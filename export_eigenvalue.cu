// mpi_merge_reader.cpp
// C++17
#include <bits/stdc++.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkXMLStructuredGridWriter.h>
#include <vtkPointData.h> 
#include <iostream>
#include "jacob.cuh"
#include "gpu_memory.cuh"
#include "gpu_macros.cuh"

using namespace std;

using Real = double;           // Fortran 側が double の想定
static constexpr bool needs_byteswap = false; // 必要なら true に

// バイトスワップ（4/8 バイト）
template<class T>
static inline void bswap_inplace(T& v) {
    auto* p = reinterpret_cast<unsigned char*>(&v);
    std::reverse(p, p + sizeof(T));
}

// Fortran unformatted sequentls ial の 1 レコード読み込み＋末尾マーカー検証
// 期待バイト数 expected_bytes のペイロードを data に読み込む。
// 先頭マーカーが 4B か 8B か自動判定（よくある 2 通りに対応）。
static void read_fortran_record(ifstream& ifs, char* data, size_t expected_bytes) {
    if (!ifs) throw runtime_error("stream not open");

    auto read_len = [&](int64_t& L, size_t nbytes) {
        ifs.read(reinterpret_cast<char*>(&L), nbytes);
        if (!ifs) throw runtime_error("failed to read record header");
        if (needs_byteswap) bswap_inplace(L);
    };

    // まず 4B ヘッダを読んでみる
    streampos pos0 = ifs.tellg();
    int32_t L4 = 0;
    ifs.read(reinterpret_cast<char*>(&L4), 4);
    if (!ifs) throw runtime_error("failed to read record header (4B)");
    if (needs_byteswap) bswap_inplace(L4);

    bool ok = false;
    if (static_cast<uint32_t>(L4) == expected_bytes) {
        // 4B マーカー方式
        ifs.read(data, expected_bytes);
        if (!ifs) throw runtime_error("failed to read record payload (4B)");
        int32_t tail = 0;
        ifs.read(reinterpret_cast<char*>(&tail), 4);
        if (!ifs) throw runtime_error("failed to read record trailer (4B)");
        if (needs_byteswap) bswap_inplace(tail);
        if (tail != L4) throw runtime_error("record trailer mismatch (4B)");
        ok = true;
    } else {
        // 8B の可能性を試す
        ifs.clear();
        ifs.seekg(pos0);
        int64_t L8 = 0;
        read_len(L8, 8);
        if (static_cast<uint64_t>(L8) != expected_bytes) {
            throw runtime_error("record length mismatch (neither 4B nor 8B)");
        }
        ifs.read(data, expected_bytes);
        if (!ifs) throw runtime_error("failed to read record payload (8B)");
        int64_t tail = 0;
        read_len(tail, 8);
        if (tail != L8) throw runtime_error("record trailer mismatch (8B)");
        ok = true;
    }
    if (!ok) throw runtime_error("unknown record format");
}

// 0 埋め文字列
static string zero_pad(long long value, int width) {
    ostringstream oss;
    oss << setw(width) << setfill('0') << value;
    return oss.str();
}

// 区間構造体（両端含む, 1-based 入力を前提）
struct Range { int sta, end; };

// 区間交差判定
static inline bool disjoint(const Range& a, const Range& b) {
    return (a.end < b.sta) || (a.sta > b.end);
}
// static inline int clamp(int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); }

// 線形化 index（Fortran と同じ i 最速）
// i in [i0, i0+ni-1], j in [j0, ...], k in [k0, ...]
static inline size_t idx3(int i, int j, int k,
                          int i0, int j0, int k0,
                          int ni, int nj, int /*nk*/) {
    return size_t(i - i0) + size_t(ni) * (size_t(j - j0) + size_t(nj) * size_t(k - k0));
}
static inline size_t idx4(int i, int j, int k, int s,
                          int i0, int j0, int k0, int s0,
                          int ni, int nj, int nk, int /*ns*/) {
    // i + ni*( j + nj*( k + nk*(s) ) ), s が最も遅い（Fortran と一致）
    return size_t(i - i0)
         + size_t(ni) * ( size_t(j - j0)
         + size_t(nj) * ( size_t(k - k0)
         + size_t(nk) * size_t(s - s0) ) );
}

// チャンネル名を定義
static const vector<string> channelNames = {
    "lambda_e"
};

vector<Real> read_coord(const string& fname, int n, int bd) {
    vector<Real> g(n + 2*bd);
    ifstream ifs(fname);
    if (!ifs) throw runtime_error("Cannot open " + fname);
    for (int i = -bd; i < n+bd; i++) {
        int idx; Real u, gi, du, dgi;
        ifs >> idx >> u >> gi >> du >> dgi;
        g[i+bd] = gi;
    }
    return g;
}

void write_vts(const string& filename,
               const vector<Real>& sf,
               int NX, int NY, int NZ,
               int x0, int y0, int z0,
               const vector<Real>& xg,
               const vector<Real>& yg,
               const vector<Real>& zg,
               int nch)
{
    // StructuredGrid を作成
    auto grid = vtkSmartPointer<vtkStructuredGrid>::New();
    grid->SetDimensions(NX, NY, NZ);

    // 座標点を登録
    auto pts = vtkSmartPointer<vtkPoints>::New();
    pts->SetDataTypeToDouble();
    pts->SetNumberOfPoints(NX * NY * NZ);

    size_t idx = 0;
    for (int k = 0; k < NZ; ++k) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                double xx = xg[x0-1 + i]; // 1-based→0-based
                double yy = yg[y0-1 + j];
                double zz = zg[z0-1 + k];
                pts->SetPoint(idx++, xx, yy, zz);
            }
        }
    }
    grid->SetPoints(pts);

    // 各チャンネルを PointData として追加
    for (int c = 0; c < nch; ++c) {
        auto arr = vtkSmartPointer<vtkDoubleArray>::New();
        string name = (c < (int)channelNames.size()) ? channelNames[c] : ("var"+to_string(c+1));
        arr->SetName(name.c_str());
        arr->SetNumberOfComponents(1);
        arr->SetNumberOfTuples(NX * NY * NZ);

        size_t npts = NX*NY*NZ;
        for (size_t n = 0; n < npts; ++n) {
            // sf は i最速, j, k, channel最遅
            size_t idx = n + size_t(NX*NY*NZ) * size_t(c);
            arr->SetValue(n, sf[idx]);
        }
        grid->GetPointData()->AddArray(arr);
    }

    // 書き出し
    auto writer = vtkSmartPointer<vtkXMLStructuredGridWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(grid);
    writer->SetDataModeToBinary(); // バイナリ出力
    writer->Write();
}

extern "C" {
    void dgeev_(char* jobvl, char* jobvr, int* n,
                double* a, int* lda,
                double* wr, double* wi,
                double* vl, int* ldvl,
                double* vr, int* ldvr,
                double* work, int* lwork, int* info);
}

__global__ void call_eval_jacob_multi(
    double t,
    const double *p_all,   // [Nsystem][NSP]
    const double *y_all,   // [Nsystem]
    double *jac_all,       // [Nsystem][NSP*NSP]
    mechanism_memory *mech_dev, int Nsystem)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nsystem) return;

    eval_jacob(t, p_all, y_all, jac_all, mech_dev);
}

int main() {
    // ======= 設定（必要に応じて実行時引数や設定ファイル化してください） =======
    const int nx = /* 全体 x サイズ */ 2000;
    const int ny = /* 全体 y サイズ */ 1100;
    const int nz = /* 全体 z サイズ */ 1;

    const int iprocs = 50, jprocs = 20, kprocs = 1; // プロセス分割
    const int ibd = 4, jbd = 4, kbd = 4;          // ハロー幅
    const int nf  = 33;                            // y の種数（>=19 を想定）

    // 出力ウィンドウ（Fortran では 1 始まり。ここも 1 始まりで指定）
    // const Range XR{1, 2000}, YR{1, 550}, ZR{1, nz};
    const Range XR{1, 200}, YR{400, 550}, ZR{1, nz};

    // 読むステップ範囲（Fortran の step0:step2:step1 に相当）
    const int step0 = 900100, step1 = 900100, step2 = 1000; // 例：単一ステップ

    // 出力配列（ウィンドウだけ確保）。C++ は 0 始まりに直す。
    const int x0 = XR.sta; const int x1 = XR.end;  // 1-based
    const int y0 = YR.sta; const int y1 = YR.end;
    const int z0 = ZR.sta; const int z1 = ZR.end;
    const int NX = x1 - x0 + 1;
    const int NY = y1 - y0 + 1;
    const int NZ = z1 - z0 + 1;

    vector<Real> xg = read_coord("../../../xg.dat", nx, ibd);
    vector<Real> yg = read_coord("../../../yg.dat", ny, jbd);
    vector<Real> zg = read_coord("../../../zg.dat", nz, kbd);

    vector<Real> p_global(size_t(NX) * NY * NZ, Real(0));
    vector<Real> sf(size_t(NX) * NY * NZ * NSP, Real(0));

    auto sfind = [&](int i, int j, int k, int c) -> size_t {
        // i,j,k は 1-based（グローバル座標）、c は 1-based チャネル番号
        // C++ 配列は 0-based に変換
        int ii = i - x0; int jj = j - y0; int kk = k - z0; int cc = c - 1;
        // i 最速, 次に j, 次に k, 最後に channel
        return size_t(ii) + size_t(NX) * ( size_t(jj) + size_t(NY) * ( size_t(kk) + size_t(NZ) * size_t(cc) ) );
    };

    auto pind = [&](int i, int j, int k) -> size_t {
        // i,j,k は 1-based（グローバル座標）
        // C++ 配列は 0-based に変換
        int ii = i - x0; int jj = j - y0; int kk = k - z0;
        // i 最速, 次に j, 次に k, 最後に channel
        return size_t(ii) + size_t(NX) * ( size_t(jj) + size_t(NY) * size_t(kk) );
    };

    // ======= ステップループ =======
    for (int step = step0; step <= step1; step += step2) {
        string filenumber = zero_pad(step, 8);

        const int nprocs = iprocs * jprocs * kprocs;
        for (int myrank = 0; myrank < nprocs; ++myrank) {
            // rank -> (i,j,k) ブロック座標（Fortran の m カウントに対応）
            int myrank_i =  myrank % iprocs;
            int myrank_j = (myrank / iprocs) % jprocs;
            int myrank_k =  myrank / (iprocs * jprocs);

            // サブドメイン範囲（1-based, 両端含む）
            int ista = nx / iprocs * myrank_i + 1;
            int iend = nx / iprocs * (myrank_i + 1);
            int jsta = ny / jprocs * myrank_j + 1;
            int jend = ny / jprocs * (myrank_j + 1);
            int ksta = nz / kprocs * myrank_k + 1;
            int kend = nz / kprocs * (myrank_k + 1);

            // ローカル（ハロー含む）配列の 1-based 範囲
            int li0 = ista - ibd, li1 = iend + ibd;
            int lj0 = jsta - jbd, lj1 = jend + jbd;
            int lk0 = ksta - kbd, lk1 = kend + kbd;

            // Range IR{li0, li1}, JR{lj0, lj1}, KR{lk0, lk1};

            // 出力ウィンドウと交差しないならスキップ
            if ( disjoint(Range{ista, iend}, XR) ||
                 disjoint(Range{jsta, jend}, YR) ||
                 disjoint(Range{ksta, kend}, ZR) ) {
                continue;
            }

            // ローカルサイズ
            int lni = li1 - li0 + 1;
            int lnj = lj1 - lj0 + 1;
            int lnk = lk1 - lk0 + 1;

            // ファイルを開く
            string cpunumber = zero_pad(myrank, 5);
            string path;
            if (nprocs != 1)
                path = string("../../") + cpunumber + "/f" + filenumber + ".dat";
            else
                path = string("../../f") + filenumber + ".dat";

            ifstream ifs(path, ios::binary);
            if (!ifs) {
                cerr << "Cannot open file: " << path << "\n";
                return 1;
            }
            cerr << "Reading " << path << "\n";

            // 読み込みバッファを確保（Fortran 配列の線形順序と一致させる）
            const size_t n3 = size_t(lni) * lnj * lnk;
            const size_t n4 = n3 * size_t(nf);
            vector<Real> u(n3), v(n3), w(n3), r(n3), p(n3), t(n3), h(n3);
            vector<Real> y(n4);

            auto read_vec = [&](vector<Real>& a) {
                const size_t bytes = a.size() * sizeof(Real);
                read_fortran_record(ifs, reinterpret_cast<char*>(a.data()), bytes);
                if (needs_byteswap) {
                    for (auto& val : a) bswap_inplace(val);
                }
            };

            read_vec(u); read_vec(v); read_vec(w); read_vec(r);
            read_vec(p); read_vec(t); read_vec(h);
            // y は 4 次元 (i,j,k,s) を一括で 1 レコードとして書いている想定
            read_vec(y);
            ifs.close();

            // マージ（ウィンドウ＆ハローでクリップ）
            const int ii_sta = max(li0, XR.sta);
            const int ii_end = min(li1, XR.end);
            const int jj_sta = max(lj0, YR.sta);
            const int jj_end = min(lj1, YR.end);
            const int kk_sta = max(lk0, ZR.sta);
            const int kk_end = min(lk1, ZR.end);

            for (int k = kk_sta; k <= kk_end; ++k)
            for (int j = jj_sta; j <= jj_end; ++j)
            for (int i = ii_sta; i <= ii_end; ++i) {
                size_t L3 = idx3(i, j, k, li0, lj0, lk0, lni, lnj, lnk);

                // 基本物理量
                // sf[sfind(i,j,k, 1)] = u[L3];
                // sf[sfind(i,j,k, 2)] = v[L3];
                // sf[sfind(i,j,k, 3)] = w[L3];
                // sf[sfind(i,j,k, 4)] = r[L3];
                // sf[sfind(i,j,k, 5)] = p[L3];
                // sf[sfind(i,j,k, 6)] = t[L3];
                // sf[sfind(i,j,k, 7)] = h[L3];
                
                p_global[pind(i,j,k)] = p[L3];

                // 種（Fortran では 1 始まりの添字）
                auto LY = [&](int s1based)->Real {
                    size_t L4 = idx4(i, j, k, s1based, li0, lj0, lk0, 1, lni, lnj, lnk, nf);
                    return y[L4];
                };

                //  Tamaoki mechanism
                sf[sfind(i,j,k,1 )] = t[L3];  // T
                sf[sfind(i,j,k,2 )] = LY(3);  // HE
                sf[sfind(i,j,k,3 )] = LY(2);  // AR
                sf[sfind(i,j,k,4 )] = LY(8);  // H2
                sf[sfind(i,j,k,5 )] = LY(5);  // O2
                sf[sfind(i,j,k,6 )] = LY(4);  // H
                sf[sfind(i,j,k,7 )] = LY(6);  // O
                sf[sfind(i,j,k,8 )] = LY(7);  // OH
                sf[sfind(i,j,k,9 )] = LY(10); // HO2
                sf[sfind(i,j,k,10)] = LY(9);  // H2O
                sf[sfind(i,j,k,11)] = LY(11); // H2O2
                sf[sfind(i,j,k,12)] = LY(33); // OHD-OH
                sf[sfind(i,j,k,13)] = LY(18); // N
                sf[sfind(i,j,k,14)] = LY(12); // NH3
                sf[sfind(i,j,k,15)] = LY(13); // NH2
                sf[sfind(i,j,k,16)] = LY(14); // NH
                sf[sfind(i,j,k,17)] = LY(20); // NNH
                sf[sfind(i,j,k,18)] = LY(19); // NO
                sf[sfind(i,j,k,19)] = LY(23); // N2O
                sf[sfind(i,j,k,20)] = LY(15); // HNO
                sf[sfind(i,j,k,21)] = LY(17); // HON
                sf[sfind(i,j,k,22)] = LY(16); // H2NO
                sf[sfind(i,j,k,23)] = LY(25); // HNOH
                sf[sfind(i,j,k,24)] = LY(24); // NH2OH
                sf[sfind(i,j,k,25)] = LY(22); // NO2
                sf[sfind(i,j,k,26)] = LY(21); // HONO
                sf[sfind(i,j,k,27)] = LY(27); // HNO2
                sf[sfind(i,j,k,28)] = LY(28); // NO3
                sf[sfind(i,j,k,29)] = LY(26); // HONO2
                sf[sfind(i,j,k,30)] = LY(29); // N2H2
                sf[sfind(i,j,k,31)] = LY(30); // H2NN
                sf[sfind(i,j,k,32)] = LY(32); // N2H4
                sf[sfind(i,j,k,33)] = LY(31); // N2H3
                // sf[sfind(i,j,k,34)] = LY(1);  // N2
            }
        }

        // ここで sf を使う（ファイル出力など）
        cout << "sf size (elements) = " << sf.size() << "\n";
        
        const int Nsystem = NX * NY * NZ;
        const double t = 0.0;
        // const double pres = 101325.0;

        // prepare parameter for CUDA
        const int threads = 256;
        // const int threads = 1024;
        const int blocks = (Nsystem + threads - 1) / threads;
        const int stride = blocks * threads; // GRID_DIM
        std::cout << "Nsystem = " << Nsystem << ", threads = " << threads << ", blocks = " << blocks << std::endl;
        
        // parameter for shared memory
        int k_max = 3;
        size_t shared_doubles_per_block = (size_t)(k_max + 1) * threads;
        size_t shared_bytes = shared_doubles_per_block * sizeof(double);
        
        // p_hostを確保して，値を代入
        double *p_host = (double*)malloc(stride * sizeof(double));
        if (!p_host) {
            fprintf(stderr, "Host memory allocation failed\n");
            exit(1);
        }
        for (int n = 0; n < stride; ++n) {
            if (n < Nsystem) {
                // p_host[n] = pres;
                p_host[n] = p_global[n];
            } else {
                p_host[n] = 0.0;
            }
        }
        // for (int n = 0; n < 16; ++n) { 
        //     printf("T_ID=%d, pres=%g\n", n, p_host[n]);
        // }

        // y_hostをNsystem*NSPで確保し、各NSPごとにy_host_singleをコピー
        double *y_host = (double*)malloc(stride * NSP * sizeof(double));
        if (!y_host) {
            fprintf(stderr, "Host memory allocation failed\n");
            exit(1);
        }
        // sf (size: NX*NY*NZ*NSP) の値を y_host (size: stride*NSP) にコピー
        // y_hostは [kk*stride + n] の順で、kk:化学種(0～NSP-1), n:空間点(0～Nsystem-1)
        for (int kk = 0; kk < NSP; ++kk) {
            for (int n = 0; n < stride; ++n) {
                if (n < Nsystem) {
                    y_host[kk * stride + n] = sf[n + Nsystem * kk];
                } else {
                    y_host[kk * stride + n] = 0.0;
                }
            }
        }

        double* jac_host = (double*)malloc(stride * NSP * NSP * sizeof(double));
        if (!jac_host) {
            fprintf(stderr, "Host memory allocation failed\n");
            exit(1);
        }
        mechanism_memory *mech_host;  // host 側ポインタ（中身は device ポインタを保持するだけ）
        mech_host = (mechanism_memory*)malloc(sizeof(mechanism_memory));

        // デバイスメモリ確保
        double *p_dev, *y_dev, *jac_dev;
        mechanism_memory *mech_dev;   // device 側の mechanism_memory 構造体
        
        // 初期化（メモリ確保＋構造体コピー）
        initialize_gpu_memory(stride, &mech_host, &mech_dev);
        cudaErrorCheck(cudaMalloc((void**)&p_dev, stride * sizeof(double)));
        cudaErrorCheck(cudaMalloc((void**)&y_dev, stride * NSP * sizeof(double)));
        cudaErrorCheck(cudaMalloc((void**)&jac_dev, stride * NSP * NSP * sizeof(double)));
        
        // ホスト → デバイス 転送
        cudaErrorCheck(cudaMemcpy(p_dev, p_host, stride * sizeof(double), cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMemcpy(y_dev, y_host, stride * NSP * sizeof(double), cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMemcpy(jac_dev, jac_host, stride * NSP * NSP * sizeof(double), cudaMemcpyHostToDevice));

        // CUDAイベントによるカーネル実行時間計測
        cudaEvent_t start, stop;
        float milliseconds = 0.0f;
        cudaErrorCheck(cudaEventCreate(&start));
        cudaErrorCheck(cudaEventCreate(&stop));
        cudaErrorCheck(cudaEventRecord(start));

        // カーネル呼び出し
        call_eval_jacob_multi<<<blocks, threads, shared_bytes>>>(t, p_dev, y_dev, jac_dev, mech_dev, Nsystem);

        cudaErrorCheck(cudaEventRecord(stop));
        cudaErrorCheck(cudaEventSynchronize(stop));
        cudaErrorCheck(cudaEventElapsedTime(&milliseconds, start, stop));
        printf("CUDA kernel execution time: %.3f ms\n", milliseconds);

        cudaErrorCheck(cudaEventDestroy(start));
        cudaErrorCheck(cudaEventDestroy(stop));

        // デバイス → ホスト 転送
        cudaErrorCheck(cudaMemcpy(jac_host, jac_dev, stride * NSP * NSP * sizeof(double), cudaMemcpyDeviceToHost));


        // 固有値計算用の変数
        double wr[NSP], wi[NSP];  // 実部と虚部
        double vl[NSP * NSP];  // 左固有ベクトル（不要ならNULLでも可）
        double vr[NSP * NSP];  // 右固有ベクトル（不要ならNULLでも可）
        int lwork = 4 * NSP;  // 作業配列のサイズ
        double work[4 * NSP];  // 作業配列
        int info;
        
        char jobvl = 'N';  // 左固有ベクトルは計算しない
        char jobvr = 'N';  // 右固有ベクトルは計算しない
        int n = NSP;
        int lda = NSP;
        int ldvl = NSP;
        int ldvr = NSP;
    
        vector<Real> eigvals(size_t(Nsystem), Real(0));

        // calculate eigenvalue for Nsystem
        for (int sys = 0; sys < Nsystem; sys++) {
            
            // pick single jacobian
            double jac_host_single[NSP*NSP] = {0.0};
            for (int r = 0; r < NSP; r++) {
                for (int c = 0; c < NSP; c++) {
                    jac_host_single[r*NSP + c] = jac_host[ (r*NSP+c)*stride + sys];
                }
            }

            // // Debugging prints
            // std::cout << "Jacobian matrix (diagonal elements):" << std::endl;
            // for (int i = 0; i < NSP; i++) {
            //     std::cout << jac_host_single[i + i*NSP] << " ";
            // }
            // std::cout << std::endl;

            // LAPACK の dgeev を呼び出し（ヤコビアンの固有値を計算）
            dgeev_(&jobvl, &jobvr, &n, jac_host_single, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
            
            if (info == 0) {
                // wrの最大値を出力
                double wr_max = wr[0];
                for (int i = 1; i < NSP; i++) {
                    // printf("%e + %ei\n", wr[i], wi[i]);
                    if (wr[i] > wr_max) wr_max = wr[i];
                }
                // printf("wr max = %e\n", wr_max);
                // std::cout << sys << "-th maximum eigenvalue = " << wr_max << std::endl;

                // sfのNSP番目の化学種にwr_maxを割り当てる
                eigvals[sys] = wr_max;

            } else {
                printf("固有値計算に失敗しました (info = %d)\n", info);
            }
        }
        
        string outname = "output_step_" + filenumber + ".vts";
        write_vts(outname, eigvals, NX, NY, NZ, x0, y0, z0, xg, yg, zg, 1);
        cerr << "Wrote " << outname << "\n";
        
        // GPUメモリ解放
        free_gpu_memory(&mech_host, &mech_dev);
        cudaErrorCheck(cudaFree(y_dev));
        cudaErrorCheck(cudaFree(p_dev));
        cudaErrorCheck(cudaFree(jac_dev));

        // CPUメモリ解放
        free(y_host);
        free(p_host);
        free(jac_host);
        free(mech_host);
    }
    
    return 0;
}


