/*
Proyecto Final: Metaheurísticas en Paralelo
Por: Guillermo Alberto Jimenez Frias
A: 03 de Junio de 2024

Compilación:        g++ -O3 -fopenmp proyecto.cpp -o main
Ejecución:          main.exe

*/

// // ================ Parametros ==================
// double Eo_inf = 0.0;    // parametros del modelo
// double Eo_sup = 0.5;
// double b_inf = 0.0;
// double b_sup = 0.2;
// double Re_inf = 0.0;
// double Re_sup = 1.0;
// double C1_inf = 0.0;
// double C1_sup = 1.14;
// double C2_inf = 0.0;
// double C2_sup = 20.8;

// double x[10] = {0.005019, 0.010038, 0.015, 0.019962, 0.0258365, 0.030057, 0.0317681, 0.0319962, 0.0330228, 0.0337072 };
// double y[10] = {0.69844, 0.6125, 0.58281, 0.5375, 0.47344, 0.36563, 0.31094, 0.2625, 0.2, 0.11094}; 


#define M_PI 3.1415926535

#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <random>
#include <omp.h>

using namespace std;

// funciones benchmark
double Sinoidal(double *x);
double Michealwicz(double *x);
double Rosenbrock(double *x);
double SixHum_Camel_Back(double *x);

// funciones objetivo
double SSE( double *y_est, double *y_real, int N );
double SAE( double *y_est, double *y_real, int N );
double MAE( double *y_est, double *y_real, int N );

// distribución uniforme y normal
void getUniform(double *x, double Linf, double Lsup, int N);
void getNormal( double *x, double mean, double std, int N, double Linf, double Lsup);
// obtener muestras de tamaños N de las distribucione anteriores
void GenUnifSample(double* Sample, int N, double *Dom, int K);
void GenNormalSample( double* Sample, int N, double* mean, double* std, double *Dom, int K );

void printVector(double *x, int N);
void printMatrix(double *Mat, int rows, int cols);
void save_matrix(char *name, double *M, int rows, int cols);
void save_matrixV00(char *name, double *M, int rows, int cols);

// metodos de ordenamiento
void SelectionSort(double arr[], int N);
void SelectionSortPair(int index[], double values[], int N);
void ShellSort(double A[], int N);
void ShellSortPair(int index[], double A[], int N);

// funciones principales
// funcion para evaluar las poblaciones
void Evaluate( double* Sample, double* y, int M, int K );
// funcion para seleccionar los mejores individuos
void BestSpecimen( double* Sample, double* y, double* BestSample, int M, int K, int M_hat );
// funcion para unir los mejores individuos como una nueva poblacion
void JointBestDescend( double* BestDesc_xi, double* Eval_xi, double* GlobalBest, double* GlobalEval, int D_best, int K );

// semilla para generar numeros aleatorios
mt19937& getGenerator() {
    static mt19937 gen( time(NULL) );
    return gen;
}

// ===================== Código Principal =============================
int main(int argc, char* argv[])
{
    omp_set_num_threads(4);

    int N = 4000;              // population size
    int K = 5;              // number or parameters
    int M = 500;             // number of descendants
    double best_rate = 0.3; // rate of best descendent
    int D_best = (int)( best_rate*M );  // best of descendent
    int L = 20;             // No of generations

    if( D_best * N < N ) 
    {
        printf("Error: Taza de Mejores muy pequeña...\n");
        return 1;
    }
    char name_file[1000];
    int MaxBest = 50;
    
    // definicion de dominio
    double Eo_inf = 0.0;    // parametros del modelo
    double Eo_sup = 0.5;
    double b_inf = 0.0;
    double b_sup = 0.2;
    double Re_inf = 0.0;
    double Re_sup = 1.0;
    double C1_inf = 0.0;
    double C1_sup = 1.14;
    double C2_inf = 0.0;
    double C2_sup = 28.0;
    double Dom[2*K] = {Eo_inf, Eo_sup, b_inf, b_sup, Re_inf, Re_sup, C1_inf, C1_sup, C2_inf, C2_sup };
    double Std_dev[K] = { 0.01*(Eo_sup), 0.01*(b_sup), 0.01*(Re_sup), 0.01*(C1_sup), 0.01*C2_sup };

    // variables
    double* xi = (double *)malloc( K * sizeof(double) );
    double* D_Pop = (double *)malloc( N * K * sizeof(double) );
    double* Eval_D_pop = ( double *)malloc( N * sizeof(double) );
    double* Desc_xi = (double *)malloc( M * K * sizeof(double) );
    double* Eval_xi = (double *)malloc( M * sizeof(double) );
    double* BestDesc_xi = (double *)malloc( D_best * K * sizeof(double) );
    double* BestDesc = (double *)malloc( (D_best + 1) * N * K * sizeof(double) );
    double* Eval_BestDesc = (double *)malloc( (D_best + 1) * N * sizeof(double) );

    auto start = chrono::high_resolution_clock::now();

    // Poblacion Inicial
    GenUnifSample( D_Pop, N, Dom, K);

    printf("Initial Population: \n");
    printMatrix( D_Pop, 5, K );     /// <----------- N != 5
    printf("\n");

    // sprintf(name_file, "Initial_Pop.txt");
    // save_matrix(name_file, D_Pop, N, K );
    
    for( int gen = 0; gen < L; gen++ )
    {
    #pragma omp parallel for firstprivate(N, M, K, D_best)
        for( int i = 0; i < N; i++ )
        {
            GenNormalSample( Desc_xi, M, D_Pop + i*K, Std_dev, Dom, K );
            // printf("Desc_x%d: \n", i);
            // printMatrix( Desc_xi, M, K);
            
            Evaluate( Desc_xi, Eval_xi, M, K );
            // printf("Eval_x%d : ", i);
            // printVector( Eval_xi, M );

            BestSpecimen( Desc_xi, Eval_xi, BestDesc_xi, M, K, D_best );
            // printf("BestDesc_x%d : \n", i);
            // printMatrix( BestDesc_xi, D_best, K );
            // printf("\n");

            // #pragma omp barrier

            // #pragma omp single
            // {
            JointBestDescend( BestDesc_xi, Eval_xi, BestDesc + i*D_best*K, Eval_BestDesc + i*D_best, D_best, K );
            
            // Evaluación de D_pop
            Evaluate( D_Pop, Eval_D_pop, N, K );
            JointBestDescend( D_Pop, Eval_D_pop, BestDesc + N*D_best*K, Eval_BestDesc + N*D_best, N, K );
            // }
        }

        // printf("Best of Best: \n");
        // printMatrix( BestDesc, D_best * N,  K );
        // printf("BestEvaluations: ");
        // printVector( Eval_BestDesc, D_best*N );

        BestSpecimen( BestDesc, Eval_BestDesc, D_Pop, (D_best+1)*N, K, N );
        // printf("Generation [%d]: \n", gen+1);
        // printMatrix( D_Pop, 5, K ); /// <----------- N != 5
        // printf("\n");

        
        // sprintf(name_file, "BestSolutions_%d.txt", gen+1);
        // save_matrix(name_file, BestDesc, MaxBest, K );
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;
    double totalTime = elapsed_seconds.count();

    // Ultimos Mejores
    printf("Generation [%d]: \n", N);
    printMatrix( D_Pop, 5, K );
    printf("\n");

    printf("Total Time: %lf\n", totalTime);

    free(D_Pop); free(Eval_D_pop);
    free(xi); free(Desc_xi); 
    free(Eval_xi); free(BestDesc_xi);
    free(BestDesc); free(Eval_BestDesc);

    return 0;
}


// funciones benchmark
double Sinoidal(double *x)
{
    /* 
    Dominio: [-2,2]x[-2,2]
    Minimo: f(0,0) = 0.1
    */
    double t1 = x[0]*x[0] + x[1]*x[1];
    double t2 = sin(5*x[0]);
    double t3 = sin(5*x[1]);
    double t = t1 + sqrt(t2*t2 + t3*t3) + 0.1;

    return t;
}

double Michealwicz(double *x)
{
    /* 
    Dominio: [0,4]x[0,4]
    Minimo: f(2.20319,1.57049) = - 1.801
    */
       
    double t1 = -sin( x[0] );
    double t2 = sin( (x[0]*x[0]) / M_PI );
    double t3 = pow( t2, 20.0 );

    double r1 = -sin( x[1] );
    double r2 = sin( (2*x[1]*x[1]) / M_PI );
    double r3 = pow( r2, 20.0 );

    return ( t1*t3 + r1*r3 );
}

double Rosenbrock(double *x)
{
    /*
    Dominio: [0,0]x[4,4]
    Minimo: f(1,1) = 0
    */
    double t1 = (1.0 - x[0]);
    double t2 = x[1] - x[0]*x[0];

    return t1*t1 + 100.0*t2*t2;
}

double SixHum_Camel_Back(double *x)
{
    /*
    Dominio: [-3,3]x[-2,2]
    Minimo: f(0,0898, −0,7126) = f(−0,0898, 0,7126) = −1.0316
    */
    double t1 = x[0]*x[0];
    double r1 = x[1]*x[1];

    double t2 = 4.0 - 2.1*t1 + 0.5*t1*t1;
    double r2 = 4.0*( r1 - 1.0 )*r1;

    return t2*t1 + x[0]*x[1] + r2;
}

// numeros aleatorios uniformes
void getUniform(double *x, double Linf, double Lsup, int N)
{
    // double L = Lsup - Linf;
    
    // uniform_real_distribution<double> dist(Linf, Lsup);

    // for( int i = 0; i < N; i++ )
    // {
    //     x[i] =  dist(getGenerator()); //(double)rand() / RAND_MAX;// + Linf;
    // }

    // for( int i = 0; i < N; i++ ){
    //     x[i] = Linf + L*((double) rand() / RAND_MAX);
    // }

    // random_device rd;
    
    uniform_real_distribution<double> UniformDist(Linf, Lsup);
    for(int i = 0; i < N; i++)
    {
        x[i] = UniformDist(getGenerator());
    }

    return;
}

void getNormal( double *x, double mean, double std_dev, int N, double Linf, double Lsup)
{
    // mt19937_64 gen( time(NULL) );
    normal_distribution<double> NormalDist( mean, std_dev );

    int count = 0;
    double number;

    while( count < N ){
        number = NormalDist(getGenerator());
        if( (Linf < number) && (number < Lsup) ){
            x[count] = number;
            count++;
        }
    }
}

void printVector(double *x, int N)
{
    int Nm1 = N - 1;
    printf("[");
    for( int i = 0; i < Nm1; i++ )
    {
        printf("%lf, ", x[i]);
    }
    printf("%lf ]\n", x[Nm1]);

    return;
}

void printMatrix(double *Mat, int rows, int cols)
{
    int idx;
    int colsm1 = cols - 1;
    printf("[");
    for( int i = 0; i < rows; i++ )
    {
        printf("[");
        for( int j = 0; j < colsm1; j++)
        {
            idx = i*cols + j;
            printf("%lf, ", Mat[idx]);
        }
        printf("%lf ], \n", Mat[i*cols + colsm1]);
    }
    printf("]\n");
}

void save_matrix(char *name, double *M, int rows, int cols)
{
    FILE *file = fopen(name, "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // fprintf(file, "%d %d\n", rows, cols);
    for (int i = 0; i < cols*rows; i++) {
        fprintf(file, "%f ", M[i]);
        if( (i%cols == cols-1) && i > 1) fprintf(file, "\n");
    }

    fclose(file);
}

void save_matrixV00(char *name, double *M, int rows, int cols)
{
    FILE *file = fopen(name, "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // fprintf(file, "%d %d\n", rows, cols);
    fprintf(file, "[ ");
    for (int i = 0; i < cols*rows; i++) {
        fprintf(file, "[ ");
        fprintf(file, "%f, ", M[i]);
        if( (i%cols == cols-1) && i > 1) fprintf(file, "], \n");
    }
    fprintf(file, "]");
    
    fclose(file);
}

void GenUnifSample(double* Sample, int N, double *Dom, int K)
{
    double *Temp = (double*)malloc( N * K * sizeof(double) );
    for( int i = 0; i < K; i++ ){
        getUniform( Temp + (i*N), Dom[2*i], Dom[(2*i)+1], N );
    }

    for( int i = 0; i < N; i++ ) {
        for ( int j = 0; j < K; j++ ) {
            Sample[i*K + j] = Temp[j*N + i];
        }
    }

    free(Temp);

    return;
}

void GenNormalSample( double* Sample, int N, double* mean, double* std_dev, double *Dom, int K )
{
    double *Temp = (double*)malloc( K*N*sizeof(double) );

    for( int i = 0; i < K; i++ ){
        getNormal( Temp + i*N, mean[i], std_dev[i], N, Dom[2*i], Dom[2*i+1] );
    }

    for( int i = 0; i < K; i++ ) {
        for ( int j = 0; j < N; j++ ) {
            Sample[j*K + i] = Temp[i*N + j];
        }
    }

    free(Temp);
}


void Evaluate( double* Sample, double* y, int M, int K )
{
    double E, b, Re, C1, C2;
    double x[10] = {0.005019, 0.010038, 0.015, 0.019962, 0.0258365, 0.030057, 0.0317681, 0.0319962, 0.0330228, 0.0337072 };
    double fx_real[10] = {0.69844, 0.6125, 0.58281, 0.5375, 0.47344, 0.36563, 0.31094, 0.2625, 0.2, 0.11094};
    double* fx_hat = (double*)malloc( 10*sizeof(double) );
    double val;

    for( int i = 0; i < M; i++ )
    {

        E = (Sample + i*K)[0];
        b = (Sample + i*K)[1];
        Re = (Sample + i*K)[2];
        C1 = (Sample + i*K)[3];
        C2 = (Sample + i*K)[4];
        
        for( int j = 0; j < 10; j++ )
        {
            val = x[j];
            fx_hat[j] = E - b*log10(val) - Re*val + C1*log(1.0 - C2*val);
        }

        y[i] = SAE( fx_hat, fx_real, 10);
    }


    free(fx_hat);
    return;
}

void BestSpecimen( double* Sample, double* y, double* BestSample, int M, int K, int M_hat )
{
    int* index = (int*)malloc( M * sizeof(int) );
    int Mm1 = M - 1;
    int L = M - M_hat - 1;
    int j;

    for( int i = 0; i < M; i++ ) { index[i] = i; }

    // SelectionSortPair( index, y, M );
    ShellSortPair( index, y, M );

    for( int i = 0; i < M_hat; i++ )
    {
        for( int j = 0; j < K; j++ )
        {
            BestSample[i*K + j] =  Sample[index[i]*K + j];
        } 
    }

    free(index);

    return;
}

void JointBestDescend( double* BestDesc_xi, double* Eval_xi, double* GlobalBest, double* GlobalEval, int D_best, int K )
{
    for( int i = 0; i < D_best; i++ )
    {
        for( int j = 0; j < K; j++ )
        {
            GlobalBest[i*K + j] =  BestDesc_xi[i*K + j];
        }
        GlobalEval[i] =  Eval_xi[i];
    }

    return;
}


// funciones objetivo
double SSE( double *y_est, double *y_real, int N)
{
    // Sum of Square Error
    double error = 0.0;
    double t;

    for( int i = 0; i < N; i ++ )
    {
        t = y_est[i] - y_real[i];
        error += t*t; 
    }

    return error;
}

double SAE( double *y_est, double *y_real, int N )
{
    // Sum of Absolute Error
    double error = 0.0;
    
    for( int i = 0; i < N; i ++ ) {
        error += abs( y_est[i] - y_real[i] );
    }

    return error;
}

double MAE( double *y_est, double *y_real, int N )
{
    // Median Absolute Error
    double error = 0.0;
    double t;

    double *V = (double*)malloc( N * sizeof(double) );
    for( int i = 0; i < N; i ++ ) {
        V[i] = abs( y_est[i] - y_real[i] );
    }

    // ordenamos
    SelectionSort( V, N );

    // calculamos la media
    if( N%2 == 0 ) {    // par
        int k = (int)N/2;
        error = (V[k] + V[k+1]) / 2;
    }
    else{   // impar
        int k = (int)N/2;
        error = V[k];
    }

    free(V);

    return error;
}

// metodos de ordenamiento
void SelectionSort(double arr[], int N)
{
    double temp;

    for( int i = 0; i < N-1; i++)
    {
        int min_idx = i;
        for( int j = i + 1; j < N; j++ )
        {
            if( arr[j] < arr[min_idx] ){
                min_idx = j;
            }
        }

        temp = arr[min_idx];
        arr[min_idx] = arr[i];
        arr[i] = temp;
    }

    return;
}

void SelectionSortPair(int index[], double values[], int N)
{
    double temp_val;
    int temp_idx;

    for( int i = 0; i < N-1; i++)
    {
        int min_idx = i;
        for( int j = i + 1; j < N; j++ )
        {
            if( values[j] < values[min_idx] ){
                min_idx = j;
            }
        }

        temp_val = values[min_idx];
        values[min_idx] = values[i];
        values[i] = temp_val;

        temp_idx = index[min_idx];
        index[min_idx] = index[i];
        index[i] = temp_idx;
    }

    return;
}

void ShellSortPair(int index[], double A[], int N)
{
	int count = 0, j = 0, h = 0;
	double temp = 0.0;
    int temp_idx;

	for(h =N/2; h > 0; h = h/2)
	{	
		for(int i = 0; i < h; i++)
		{
			for (int f = h + i; f < N; f = f + h)
			{
				j = f;
				while(j > i && A[j-h] > A[j])
				{
					temp = A[j];
					A[j] = A[j-h];
					A[j-h] = temp;

                    temp_idx = index[j];
                    index[j] = index[j-h];
                    index[j-h] = temp_idx;

					j = j -h;
				}
			}
		}
	}
}



