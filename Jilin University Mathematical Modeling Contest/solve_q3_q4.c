#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 60
#define MAX_COLS 24
#define MAX_INTERVALS 8

static const double Q3_Y[N] = {
    1.0, 2.2, 3.4, 4.6, 34.2, 37.0, 39.7, 42.4, 45.1, 20.9,
    30.6, 39.7, 47.5, 80.7, 87.0, 92.4, 97.0, 100.6, 80.9, 83.0,
    84.0, 83.8, 109.7, 106.1, 101.2, 101.6, 102.0, 79.0, 80.2, 81.4,
    82.6, 108.6, 110.6, 112.6, 110.0, 107.4, 76.0, 71.4, 66.8, 62.2,
    76.6, 72.0, 67.4, 62.8, 62.8, 43.8, 43.8, 43.8, 43.8, 81.9,
    82.1, 82.3, 82.5, 82.7, 38.7, 37.9, 37.0, 36.2, 52.3, 51.4};

static const double Q4_Y[N] = {
    20.1177894911454, 21.6463354966050, 31.0158945303481, 40.5070068491122,
    41.4588739698033, 40.3803232399953, 45.0279857509364, 23.5750521536399,
    27.6600546834744, 28.5191082815942, 22.8609046534204, 54.4147422337912,
    56.3573355837284, 54.6330054730814, 59.2022438265944, 55.9850905871025,
    37.2740895596159, 43.2758840541349, 44.3376592504386, 46.9259386352619,
    74.7459263724498, 69.1356925038834, 74.9930341811205, 75.0219464698382,
    72.3298810063705, 52.8119207370747, 53.9450597347717, 52.3878159778148,
    55.7438234159182, 81.1194930253829, 86.6483694226477, 81.2546630023012,
    83.2153843040771, 81.3693707545236, 53.7770536778919, 59.5876630887635,
    58.5586291348608, 54.5367961611461, 80.8017757824831, 71.8755686637349,
    73.5099554395689, 68.9524675968697, 67.9241341720243, 44.8615990775196,
    36.4949814133466, 35.4181153665916, 31.5646898503186, 71.6705040637644,
    68.6749189197200, 65.5374928009125, 58.2082010430087, 60.4376210794866,
    17.2407840369575, 12.3008942351006, 10.9519809787163, 12.9869122803657,
    36.8779519139853, 31.1230856242529, 32.2821418110186, 28.5904957150662};

typedef struct {
    double score;
    double bp[5];
    int bp_count;
    double coef[MAX_COLS];
    int cols;
    double fit[N];
    double intervals[MAX_INTERVALS][2];
    int interval_count;
    double first_green_minute;
    double weights[N];
} FitResult;

static double dmin(double a, double b) { return a < b ? a : b; }
static double dmax(double a, double b) { return a > b ? a : b; }

static void make_clock(double t, char *buf, int n)
{
    int minutes = 7 * 60 + (int)floor(2.0 * t + 0.5);
    snprintf(buf, n, "%02d:%02d", minutes / 60, minutes % 60);
}

static void calc_metrics(const double y[], const double fit[], double *rmse, double *mae, double *maxae, double *mape)
{
    double se = 0.0, ae = 0.0, pe = 0.0, mx = 0.0;
    for (int i = 0; i < N; i++) {
        double e = fit[i] - y[i];
        double ab = fabs(e);
        se += e * e;
        ae += ab;
        if (ab > mx) mx = ab;
        pe += ab / dmax(fabs(y[i]), 1e-9);
    }
    *rmse = sqrt(se / N);
    *mae = ae / N;
    *maxae = mx;
    *mape = 100.0 * pe / N;
}

static int solve_linear_system(int n, double mat[MAX_COLS][MAX_COLS], double rhs[MAX_COLS], double sol[MAX_COLS])
{
    for (int k = 0; k < n; k++) {
        int piv = k;
        double best = fabs(mat[k][k]);
        for (int i = k + 1; i < n; i++) {
            if (fabs(mat[i][k]) > best) {
                best = fabs(mat[i][k]);
                piv = i;
            }
        }
        if (best < 1e-10) return 0;
        if (piv != k) {
            for (int j = k; j < n; j++) {
                double tmp = mat[k][j];
                mat[k][j] = mat[piv][j];
                mat[piv][j] = tmp;
            }
            double tmp = rhs[k];
            rhs[k] = rhs[piv];
            rhs[piv] = tmp;
        }
        for (int i = k + 1; i < n; i++) {
            double f = mat[i][k] / mat[k][k];
            for (int j = k; j < n; j++) mat[i][j] -= f * mat[k][j];
            rhs[i] -= f * rhs[k];
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        double s = rhs[i];
        for (int j = i + 1; j < n; j++) s -= mat[i][j] * sol[j];
        sol[i] = s / mat[i][i];
    }
    return 1;
}

static int least_squares(const double A[N][MAX_COLS], const double y[], const double w[], int cols, double coef[MAX_COLS])
{
    double normal[MAX_COLS][MAX_COLS] = {{0.0}};
    double rhs[MAX_COLS] = {0.0};
    double sol[MAX_COLS] = {0.0};
    for (int r = 0; r < N; r++) {
        double wr = w ? w[r] : 1.0;
        for (int i = 0; i < cols; i++) {
            rhs[i] += wr * A[r][i] * y[r];
            for (int j = 0; j < cols; j++) {
                normal[i][j] += wr * A[r][i] * A[r][j];
            }
        }
    }
    for (int i = 0; i < cols; i++) normal[i][i] += 1e-8;
    if (!solve_linear_system(cols, normal, rhs, sol)) return 0;
    for (int i = 0; i < cols; i++) coef[i] = sol[i];
    return 1;
}

static void predict(const double A[N][MAX_COLS], const double coef[MAX_COLS], int cols, double fit[N])
{
    for (int r = 0; r < N; r++) {
        double s = 0.0;
        for (int c = 0; c < cols; c++) s += A[r][c] * coef[c];
        fit[r] = s;
    }
}

static void q1_q3_cols(double x, const double bp[5], double out[2])
{
    double a = bp[0], b = bp[1], c = bp[2], d = bp[3], e = bp[4];
    out[0] = 0.0;
    out[1] = 0.0;
    if (a < x && x <= b) out[0] = (x - a) / (b - a);
    else if (b < x && x <= c) {
        out[0] = (c - x) / (c - b);
        out[1] = (x - b) / (c - b);
    } else if (c < x && x <= d) out[1] = 1.0;
    else if (d < x && x <= e) out[1] = (e - x) / (e - d);
}

static double q1_q4_col(double x, const double bp[4])
{
    double a = bp[0], b = bp[1], c = bp[2], d = bp[3];
    if (a < x && x <= b) return (x - a) / (b - a);
    if (b < x && x <= c) return 1.0;
    if (c < x && x <= d) return (d - x) / (d - c);
    return 0.0;
}

static void q2_cols(double x, double up_end, double flat_end, double out[3])
{
    out[0] = 1.0;
    out[1] = dmin(dmax(x + 1.0, 0.0), up_end + 1.0);
    out[2] = dmax(x - flat_end, 0.0);
}

static int signal_cols(double first_green_minute, double intervals[MAX_INTERVALS][2])
{
    int count = 0;
    double start = first_green_minute / 2.0;
    while (start < N && count < MAX_INTERVALS) {
        intervals[count][0] = start;
        intervals[count][1] = dmin(N - 1.0, start + 5.5);
        count++;
        start += 9.0;
    }
    return count;
}

static void build_q3_design(const double bp[5], double A[N][MAX_COLS], double intervals[MAX_INTERVALS][2], int interval_count)
{
    memset(A, 0, sizeof(double) * N * MAX_COLS);
    for (int r = 0; r < N; r++) {
        double x = (double)r - 1.5;
        double q1[2], q2[3];
        q1_q3_cols(x, bp, q1);
        q2_cols(x, 36.0, 48.0, q2);
        A[r][0] = q1[0];
        A[r][1] = q1[1];
        A[r][2] = q2[0];
        A[r][3] = q2[1];
        A[r][4] = q2[2];
        for (int i = 0; i < interval_count; i++) {
            double st = intervals[i][0], en = intervals[i][1];
            if ((double)r >= st && (double)r < en) {
                A[r][5 + 2 * i] = 1.0;
                A[r][6 + 2 * i] = (double)r - st;
            }
        }
    }
}

static void build_q4_design(const double bp[4], double A[N][MAX_COLS], double intervals[MAX_INTERVALS][2], int interval_count)
{
    memset(A, 0, sizeof(double) * N * MAX_COLS);
    for (int r = 0; r < N; r++) {
        double x = (double)r - 1.5;
        double q2[3];
        A[r][0] = q1_q4_col(x, bp);
        q2_cols(x, 18.0, 36.0, q2);
        A[r][1] = q2[0];
        A[r][2] = q2[1];
        A[r][3] = q2[2];
        for (int i = 0; i < interval_count; i++) {
            double st = intervals[i][0], en = intervals[i][1];
            if ((double)r >= st && (double)r < en) {
                A[r][4 + 2 * i] = 1.0;
                A[r][5 + 2 * i] = (double)r - st;
            }
        }
    }
}

static double negative_penalty(const double vals[], int n)
{
    double p = 0.0;
    for (int i = 0; i < n; i++) {
        if (vals[i] < 0.0) p += vals[i] * vals[i];
    }
    return p / n;
}

static void update_huber_weights(const double y[], const double fit[], double w[])
{
    double abs_res[N], tmp[N];
    for (int i = 0; i < N; i++) {
        tmp[i] = y[i] - fit[i];
    }
    for (int i = 0; i < N - 1; i++) {
        for (int j = i + 1; j < N; j++) {
            if (tmp[j] < tmp[i]) {
                double z = tmp[i]; tmp[i] = tmp[j]; tmp[j] = z;
            }
        }
    }
    double med = 0.5 * (tmp[29] + tmp[30]);
    for (int i = 0; i < N; i++) abs_res[i] = fabs((y[i] - fit[i]) - med);
    for (int i = 0; i < N - 1; i++) {
        for (int j = i + 1; j < N; j++) {
            if (abs_res[j] < abs_res[i]) {
                double z = abs_res[i]; abs_res[i] = abs_res[j]; abs_res[j] = z;
            }
        }
    }
    double scale = 1.4826 * 0.5 * (abs_res[29] + abs_res[30]) + 1e-6;
    double c = 1.345 * scale;
    for (int i = 0; i < N; i++) {
        double r = fabs(y[i] - fit[i]);
        w[i] = dmin(1.0, c / dmax(r, 1e-9));
    }
}

static FitResult fit_q3(void)
{
    FitResult best;
    double intervals[MAX_INTERVALS][2], A[N][MAX_COLS], coef[MAX_COLS], fit[N];
    int int_count = signal_cols(8.0, intervals);
    best.score = 1e100;

    for (double a = -1.0; a <= 4.1; a += (a < 0.0 ? 1.0 : 2.0)) {
        for (int b = (int)(a + 8); b < 25; b += 2) {
            for (int c = b + 6; c < 43; c += 2) {
                for (int d = c + 4; d < 53; d += 2) {
                    double e_list[3] = {55.0, 59.0, 61.0};
                    for (int ei = 0; ei < 3; ei++) {
                        double bp[5] = {a, (double)b, (double)c, (double)d, e_list[ei]};
                        int cols = 5 + 2 * int_count;
                        build_q3_design(bp, A, intervals, int_count);
                        if (!least_squares(A, Q3_Y, NULL, cols, coef)) continue;
                        predict(A, coef, cols, fit);

                        double rmse, mae, maxae, mape;
                        calc_metrics(Q3_Y, fit, &rmse, &mae, &maxae, &mape);
                        double sig[N];
                        for (int r = 0; r < N; r++) {
                            sig[r] = 0.0;
                            for (int k = 5; k < cols; k++) sig[r] += A[r][k] * coef[k];
                        }
                        double penalty = 0.0;
                        if (coef[0] < coef[1] || coef[1] < 0.0) penalty += 1000.0;
                        if (coef[2] < 0.0 || coef[3] < 0.0 || coef[4] > 0.0) penalty += 1000.0;
                        penalty += 25.0 * negative_penalty(sig, N) + 5.0 * negative_penalty(fit, N);
                        if (rmse + penalty < best.score) {
                            best.score = rmse + penalty;
                            best.bp_count = 5;
                            memcpy(best.bp, bp, sizeof(double) * 5);
                            best.cols = cols;
                            memcpy(best.coef, coef, sizeof(double) * cols);
                            memcpy(best.fit, fit, sizeof(double) * N);
                            best.interval_count = int_count;
                            memcpy(best.intervals, intervals, sizeof(intervals));
                            best.first_green_minute = 8.0;
                        }
                    }
                }
            }
        }
    }
    return best;
}

static FitResult fit_q4(void)
{
    FitResult best;
    double intervals[MAX_INTERVALS][2], A[N][MAX_COLS], coef[MAX_COLS], fit[N], w[N];
    best.score = 1e100;

    for (double fg = 0.0; fg < 18.0; fg += 1.0) {
        int int_count = signal_cols(fg, intervals);
        for (double a = -1.0; a <= 6.1; a += (a < 0.0 ? 1.0 : 2.0)) {
            for (int b = (int)(a + 6); b < 29; b += 2) {
                for (int c = b + 6; c < 47; c += 2) {
                    double d_list[3] = {53.0, 57.0, 61.0};
                    for (int di = 0; di < 3; di++) {
                        double bp[4] = {a, (double)b, (double)c, d_list[di]};
                        int cols = 4 + 2 * int_count;
                        build_q4_design(bp, A, intervals, int_count);
                        for (int i = 0; i < N; i++) w[i] = 1.0;
                        if (!least_squares(A, Q4_Y, w, cols, coef)) continue;
                        for (int it = 0; it < 8; it++) {
                            predict(A, coef, cols, fit);
                            update_huber_weights(Q4_Y, fit, w);
                            if (!least_squares(A, Q4_Y, w, cols, coef)) break;
                        }
                        predict(A, coef, cols, fit);

                        double weighted_se = 0.0, weight_sum = 0.0;
                        for (int r = 0; r < N; r++) {
                            double e = fit[r] - Q4_Y[r];
                            weighted_se += w[r] * e * e;
                            weight_sum += w[r];
                        }
                        double robust = sqrt(weighted_se / dmax(weight_sum, 1e-9));
                        double sig[N];
                        for (int r = 0; r < N; r++) {
                            sig[r] = 0.0;
                            for (int k = 4; k < cols; k++) sig[r] += A[r][k] * coef[k];
                        }
                        double penalty = 0.0;
                        if (coef[0] < 0.0) penalty += 1000.0;
                        if (coef[1] < 0.0 || coef[2] < 0.0 || coef[3] > 0.0) penalty += 1000.0;
                        penalty += 25.0 * negative_penalty(sig, N);
                        if (robust + penalty < best.score) {
                            best.score = robust + penalty;
                            best.bp_count = 4;
                            memcpy(best.bp, bp, sizeof(double) * 4);
                            best.cols = cols;
                            memcpy(best.coef, coef, sizeof(double) * cols);
                            memcpy(best.fit, fit, sizeof(double) * N);
                            memcpy(best.weights, w, sizeof(double) * N);
                            best.interval_count = int_count;
                            memcpy(best.intervals, intervals, sizeof(intervals));
                            best.first_green_minute = fg;
                        }
                    }
                }
            }
        }
    }
    return best;
}

static double eval_q1_q3(double t, const FitResult *r)
{
    double q1[2];
    q1_q3_cols(t, r->bp, q1);
    return q1[0] * r->coef[0] + q1[1] * r->coef[1];
}

static double eval_q1_q4(double t, const FitResult *r)
{
    return q1_q4_col(t, r->bp) * r->coef[0];
}

static double eval_q2(double t, const double coef[3], double up_end, double flat_end)
{
    double q2[3];
    q2_cols(t, up_end, flat_end, q2);
    return q2[0] * coef[0] + q2[1] * coef[1] + q2[2] * coef[2];
}

static double eval_signal(double t, const FitResult *r, int offset)
{
    for (int i = 0; i < r->interval_count; i++) {
        double st = r->intervals[i][0], en = r->intervals[i][1];
        if (st <= t && t < en) {
            return r->coef[offset + 2 * i] + r->coef[offset + 2 * i + 1] * (t - st);
        }
    }
    return 0.0;
}

static void print_fit_q3(const FitResult *r)
{
    double rmse, mae, maxae, mape;
    calc_metrics(Q3_Y, r->fit, &rmse, &mae, &maxae, &mape);
    printf("\n================ 问题3 ================\n");
    printf("支路1断点: t=(%.1f, %.0f, %.0f, %.0f, %.1f), peak=%.4f, stable=%.4f\n",
           r->bp[0], r->bp[1], r->bp[2], r->bp[3], r->bp[4], r->coef[0], r->coef[1]);
    printf("支路2: q2(t)=%.4f + %.4f*min(max(t+1,0),37) + (%.4f)*max(t-48,0)\n",
           r->coef[2], r->coef[3], r->coef[4]);
    printf("支路3绿灯分段:\n");
    for (int i = 0; i < r->interval_count; i++) {
        char s[16], e[16];
        make_clock(r->intervals[i][0], s, sizeof(s));
        make_clock(r->intervals[i][1], e, sizeof(e));
        printf("  [%s,%s): q3(t)=%.4f + %.4f*(t-%.1f); 红灯 q3(t)=0\n",
               s, e, r->coef[5 + 2 * i], r->coef[6 + 2 * i], r->intervals[i][0]);
    }
    printf("误差: RMSE=%.4f, MAE=%.4f, 最大绝对误差=%.4f, MAPE=%.2f%%\n", rmse, mae, maxae, mape);
    for (int i = 0; i < 2; i++) {
        double t = i == 0 ? 15.0 : 45.0;
        char name[16];
        make_clock(t, name, sizeof(name));
        double x = t - 1.5;
        double q1 = dmax(0.0, eval_q1_q3(x, r));
        double q2 = dmax(0.0, eval_q2(x, &r->coef[2], 36.0, 48.0));
        double q3 = dmax(0.0, eval_signal(t, r, 5));
        printf("%s: 支路1=%.4f, 支路2=%.4f, 支路3=%.4f, 合计=%.4f\n", name, q1, q2, q3, q1 + q2 + q3);
    }
}

static void print_fit_q4(const FitResult *r)
{
    double rmse, mae, maxae, mape;
    calc_metrics(Q4_Y, r->fit, &rmse, &mae, &maxae, &mape);
    printf("\n================ 问题4 ================\n");
    printf("估计首个绿灯开始时刻: 7:%02d\n", (int)floor(r->first_green_minute + 0.5));
    printf("支路1断点: t=(%.1f, %.0f, %.0f, %.1f), stable=%.4f\n",
           r->bp[0], r->bp[1], r->bp[2], r->bp[3], r->coef[0]);
    printf("支路2: q2(t)=%.4f + %.4f*min(max(t+1,0),19) + (%.4f)*max(t-36,0)\n",
           r->coef[1], r->coef[2], r->coef[3]);
    printf("支路3绿灯分段:\n");
    for (int i = 0; i < r->interval_count; i++) {
        char s[16], e[16];
        make_clock(r->intervals[i][0], s, sizeof(s));
        make_clock(r->intervals[i][1], e, sizeof(e));
        printf("  [%s,%s): q3(t)=%.4f + %.4f*(t-%.1f); 红灯 q3(t)=0\n",
               s, e, r->coef[4 + 2 * i], r->coef[5 + 2 * i], r->intervals[i][0]);
    }
    printf("误差(对含噪观测): RMSE=%.4f, MAE=%.4f, 最大绝对误差=%.4f, MAPE=%.2f%%\n", rmse, mae, maxae, mape);
    printf("稳健拟合识别的较大扰动采样点:");
    int any = 0;
    for (int i = 0; i < N; i++) {
        if (r->weights[i] < 0.5) {
            printf(" t=%d", i);
            any = 1;
        }
    }
    if (!any) printf(" 无");
    printf("\n");
    for (int i = 0; i < 2; i++) {
        double t = i == 0 ? 15.0 : 45.0;
        char name[16];
        make_clock(t, name, sizeof(name));
        double x = t - 1.5;
        double q1 = dmax(0.0, eval_q1_q4(x, r));
        double q2 = dmax(0.0, eval_q2(x, &r->coef[1], 18.0, 36.0));
        double q3 = dmax(0.0, eval_signal(t, r, 4));
        printf("%s: 支路1=%.4f, 支路2=%.4f, 支路3=%.4f, 合计=%.4f\n", name, q1, q2, q3, q1 + q2 + q3);
    }
}

int main(void)
{
    FitResult q3 = fit_q3();
    FitResult q4 = fit_q4();
    print_fit_q3(&q3);
    print_fit_q4(&q4);
    return 0;
}
