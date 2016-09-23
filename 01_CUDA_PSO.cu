/*
FUNCTION
---------------------
    SPH .   Sphere.                 Min : x_i = 1                     Bounds : [-10, 10]
    SKT .   Styblinski-Tang         Min : x_i = -2.903534             Bounds : [-5 , 5 ]
    DXP .   Dixon-Price             Min : x_i = 2^(-(2^i - 2)/(2^2))  Bounds : [-10, 10]
    RSB .   Rosenbrock              Min : x_i = 1                     Bounds : [-2.048, 2.048]
    ACK .   Ackley                  Min : x_i = 0                     Bounds : [-32.768, 32.768]
    GWK .   Griewank                Min : x_i = 0                     Bounds : [-600, 600]
    RTG .   Rastrigin               Min : x_i = 0                     Bounds : [-5.12, 5.12]
    LEV .   Levy                    Min : x_i = 1                     Bounds : [-10, 10]
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define F_KEY SPH
#define F_NAME "Sphere"

#define N 32   // Dimension
#define POPULATION 128 // Population
#define UB 10 // Upper bound
#define LB -10 // Lower bound

#define W 0.6
#define PHI_P 1.6
#define PHI_G 1.6


#define WARP_SIZE 32
#define N_WARPS N/WARP_SIZE
#define N_ITERS 1000



__device__ int iMin = 0;
__device__ int fitMin = 9999999;

// Prints an array of floats from Device
void fcudaPrint(float * array, int elements, int n_jump){
  float * aux;
  aux = (float *) malloc(elements * sizeof(float));
  cudaMemcpy(aux, array, elements * sizeof(float), cudaMemcpyDeviceToHost);

  int i;
  for (i = 0; i < elements; i++){
      if ((i % n_jump) == 0 ) {printf("\n");}
      printf("%.5f ", aux[i]);
  }
  free(aux);
  aux = NULL;
}

// Prints an array of ints from Device
void icudaPrint(int * array, int elements, int n_jump){
  int * aux;
  aux = (int *) malloc(elements * sizeof(int));
  cudaMemcpy(aux, array, elements * sizeof(int), cudaMemcpyDeviceToHost);

  int i;
  for (i = 0; i < elements; i++){
      if ((i % n_jump) == 0 ) {printf("\n");}
      printf("%d ", aux[i]);
  }
  free(aux);
  aux = NULL;
}

__global__ void init_states(unsigned int seed, curandState_t * states) {
  /* we have to initialize the state */
  curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

// Generates random uniform numbers beteen 0 and 1
__global__ void U_01(float * arr, curandState_t * states){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = curand_uniform(&states[id]);
}

// Generates random normal numbers between mu = 0, sigma = 1
__global__ void N_01(float * arr, curandState_t * states){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = curand_normal(&states[id])/10;
}

// Generates random integers beteewn 0 and n
__global__ void irand(int n, int * arr, curandState_t * states){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = curand(&states[id]) % n;
}

// Sets up the initial population
__global__ void init_pos(float * pos, float * rand_uniform){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  pos[id] = LB + rand_uniform[id] * (UB - LB);
}

// Sets up the initial velocity
__global__ void init_vel(float * vel, float * rand_uniform){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  vel[id] = LB + rand_uniform[id] * (UB - LB);
}

// Fills an array with zeros
__global__ void zeros(int * arr){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = 0;
}

// Fills an array with a number n
__global__ void ffill(float * arr, float n){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  arr[id] = n;
}

// Function to minimize : Sphere
__global__ void SPH(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;

  float value;
  int i = blockIdx.x * blockDim.x + threadIdx.x;


  // Init the sum value to 0
  if (threadIdx.x == 0) {sum = 0;}
  // The main operation
  //value = __fmul_rd(__fsub_rd(x[i],1),__fsub_rd(x[i],1));
  value = pow(x[i] - 1,2);

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }

  // The first thread of each warp adds its value
  if ((i & 31) == 0){
    atomicAdd(&sum, value);
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block-bee
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Function to minimize : Styblinski-Tang
__global__ void SKT(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;
  //__shared__ float prod;
  //__shared__ float sum2;

  float value;
  int i = blockIdx.x * blockDim.x + threadIdx.x;


  if (threadIdx.x == 0) {sum = 39.166599*N;}
  // The main operation
  //value = __fadd_ru(__fsub_ru(__fmul_ru(0.5,__powf(x[i], 4)), __fmul_ru(8,__powf(x[i],2))), __fmul_ru(2.5, x[i]));
  value = 0.5 * pow(x[i], 4) - 8 * pow(x[i],2) + 2.5*x[i];

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }

  // The first thread of each warp adds its value
  if ((i & 31) == 0){
    atomicAdd(&sum, value);
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block-bee
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Function to minimize : Dixon-Price
__global__ void DXP(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;

  float value;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0) {
    sum = 0;
    value = pow(x[0] - 1,2);
  } else {
    value = threadIdx.x * pow(2 *pow(x[i],2) - x[i-1] ,2);
  }

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }

  // The first thread of each warp adds its value
  if ((i & 31) == 0){

    atomicAdd(&sum, value);
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block-bee
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Function to minimize : Rosenbrock (esta va raro)
__global__ void RSB(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;

  float value;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == (N-1)) {
    sum = 0;
    value = 0;
  } else {
    value = 100 * pow(x[i+1] - pow(x[i],2),2) + pow(1 - x[i],2);
  }

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }

  // The first thread of each warp adds its value
  if ((i & 31) == 0){
    atomicAdd(&sum, value);
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block-bee
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Function to minimize : Zakharov
__global__ void ZKV(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;

  float value0, value1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Init the sum value to 0
  if (threadIdx.x == 0){
    sum = 0;
  }

  // The main operation
  //value = __fmul_rd(__fsub_rd(x[i],1),__fsub_rd(x[i],1));
  value0 = pow(x[i],2);
  value1 = 0.5 * threadIdx.x * (x[i]);

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value0 += __shfl_down(value0, offset);
    value1 += __shfl_down(value1, offset);
  }

  // The first thread of each warp adds its value
  if ((i & 31) == 0){
    atomicAdd(&sum, value0 + pow(value1, 2) + pow(value1, 4));
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block-bee
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Function to minimize : Ackley
__global__ void ACK(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;

  float value0, value1;
  float addend0, addend1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Some parameters
  float a = 20;
  float b = 0.2;
  float c = 2 * M_PI;

  // Init the sum value to 0
  if (threadIdx.x == 0){
    sum = 0;
  }

  // The main operation
  //value = __fmul_rd(__fsub_rd(x[i],1),__fsub_rd(x[i],1));
  value0 = pow(x[i], 2);
  value1 = __cosf(c * x[i]);
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value0 += __shfl_down(value0, offset);
    value1 += __shfl_down(value1, offset);
  }

  // Compute the addends in parallel
  if (threadIdx.x == 0){
    addend0 = -a * __expf(__fmul_rd(-b, __fsqrt_rd(__frcp_rz(N) * value0)));
    //printf("%f\n", addend0);
  } else if (threadIdx.x == 1){
    addend1 = __expf(__fmul_rd(__frcp_rz(N), value1));
    //printf("%f\n", addend1);
  }

  // The first thread of each warp adds its value
  if ((i & 31) == 0){
    atomicAdd(&sum, addend0 - addend1 + a + __expf(1));
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block-bee
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Function to minimize : Griewank
__global__ void GWK(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;

  float value0, value1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Init the sum value to 0
  if (threadIdx.x == 0){
    sum = 0;
  }

  // The main operation
  //value = __fmul_rd(__fsub_rd(x[i],1),__fsub_rd(x[i],1));
  value0 = pow(x[i], 2);
  value1 = __cosf(__fdiv_rn(x[i], __fsqrt_rn(threadIdx.x + 1)));

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value0 += __shfl_down(value0, offset);
    value1 *= __shfl_down(value1, offset);
  }

  // The first thread of each warp adds its value
  if ((i & 31) == 0){
    atomicAdd(&sum, __fdiv_rn(value0, 4000) - value1 + 1);
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Function to minimize : Rastrigin
__global__ void RTG(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;

  float value;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0) {sum = 10*N;}
  // The main operation
  //value = __fadd_ru(__fsub_ru(__fmul_ru(0.5,__powf(x[i], 4)), __fmul_ru(8,__powf(x[i],2))), __fmul_ru(2.5, x[i]));
  value = pow(x[i],2) - 10*cos(2 * M_PI * x[i]);

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }

  // The first thread of each warp adds its value
  if ((i & 31) == 0){
    atomicAdd(&sum, value);
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Function to minimize : Levy
__global__ void LEV(float * x, float * evals){
  // Launch 1 block of N threads for each element fo the population: <<<POPULATION, N>>>

  // variable sum is shared by all threads in each block
  __shared__ float sum;

  float value;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float w = 1 + ((x[i] - 1)/4);

  if (threadIdx.x == 0) {
    sum = 0; // Sets sum to 0
    value = pow(__sinf(__fmul_ru(M_PI, w)), 2) + pow(w - 1, 2) * (1 + 10 * pow(__sinf(M_PI * w + 1),2));
  } else if (threadIdx.x == (N-1)){
    value = pow(w - 1, 2) * (1 + pow(__sinf(2*M_PI * w),2));
  } else {
    value = pow(w - 1, 2) * (1 + 10 * pow(__sinf(M_PI * w + 1),2));
  }

  // A little warp reduction in order to reduce the amount of atomic operations
  int offset;
  for (offset = WARP_SIZE/2; offset>0; offset >>= 1){
    value += __shfl_down(value, offset);
  }

  // The first thread of each warp adds its value
  if ((i & 31) == 0){
    atomicAdd(&sum, value);
  }

  // Thread synchronization, because this is not a warp operation
  __syncthreads();

  // Only one thread writes the result of this block-bee
  if (threadIdx.x == 0){
      evals[blockIdx.x] = sum;
  }
}

// Updates the min array with the actual position of each particle
__global__ void update_min(float * pos, float * min, float * evals, float * min_evals){
  // Needed to lauch as many blocks as particles, with N threads per block
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.x;

  // We're using more threads than needed, but we maximize the GPU usage
  if (evals[i] < min_evals[i]){
    min_evals[i] = evals[i];
    min[id] = pos[id];
  }
}

// Updates the gMin
__global__ void update_gMin(float * min, float * min_evals, float * gMin){
  // It's necessary launch as many threads as dim. In general, 1 block of N threads will be enough
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (min_evals[iMin] < fitMin){
    gMin[id] = min[iMin * N + id];
  }
}


// Updates the velocity
__global__ void update_vel(float * vel, float * pos, float * min, float * gMin, float * ru01, float * ru02){
  // It's necessary launch as many blocks of N threads as particles
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int i = threadIdx.x;

  vel[id] = W * vel[id] + PHI_P * ru01[id] * (min[id] - pos[id]) +  PHI_G * ru02[id] * (gMin[i] - pos[id]);
}

// Updates the position
__global__ void update_pos(float * pos, float * vel){
  // It's necessary launch as many blocks of N threads as particles
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  pos[id] = pos[id] + vel[id];
}

// Operación atómica que escribe el mínimo de un índice
__device__ float atomicMinIndex(float * array, int * address, int val){
  int lo_que_tengo, lo_que_tenia;
  lo_que_tengo = * address;
  //printf("Esto : %f debe ser menor que esto %f\n", array[val], array[lo_que_tengo]);
  while (array[val] < array[lo_que_tengo]){
    lo_que_tenia = lo_que_tengo;
    lo_que_tengo = atomicCAS(address, lo_que_tenia, val);
  }
  return lo_que_tengo;
}


//  In this case we don't use the influence of the global minima. In addition
//  a member is only replaced by a better solution, so we can only use this
//  step at the end of the algorithm, unless we use this as stop criterion
__global__  void arrayReduction(float * array){
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int thisThreadId = id;
  float value = array[id];
  int gap, id2;
  float value2;
  for (gap = WARP_SIZE/2; gap > 0; gap >>= 1){
    id2 = __shfl_down(id, gap);
    value2 = __shfl_down(value, gap);
    if (value2 < value){
      value = value2;
      id = id2;
    }
  }
  if (((thisThreadId & (WARP_SIZE - 1)) == 0)){
    atomicMinIndex(array, &iMin, id);
  }
}


__global__ void print_iMin(float * pos){
  int i;
  for (i = 0; i < N; i++){
    printf("%f ", pos[iMin * N + i]);
  }
}

__global__ void print_gMin(float * gMin){
  int i;
  printf("\n");
  for (i = 0; i < N; i ++){
    printf("%f ", gMin[i]);
  }
  printf("\n");
}

int main(void){

  // States
  curandState_t * states;
  cudaMalloc((void**) &states, N * POPULATION * sizeof(curandState_t));
  init_states<<<N*POPULATION,1>>>(time(0), states);

  // The random things
  float * ru01, * ru02;

  cudaMalloc((void **) &ru01, N * POPULATION *sizeof(float));
  cudaMalloc((void **) &ru02, N * POPULATION *sizeof(float));
  U_01<<<N, POPULATION>>>(ru01, states);
  U_01<<<N, POPULATION>>>(ru02, states);

  // All the stuff
  float * pos, * min, * vel;
  float * evals, * min_evals;
  float * gMin;

  cudaMalloc((void **) &pos, N * POPULATION * sizeof(float));
  cudaMalloc((void **) &min, N * POPULATION * sizeof(float));
  cudaMalloc((void **) &vel, N * POPULATION * sizeof(float));
  cudaMalloc((void **) &evals, POPULATION * sizeof(float));
  cudaMalloc((void **) &min_evals, POPULATION * sizeof(float));
  cudaMalloc((void **) &gMin, N * sizeof(float));

  init_pos<<<POPULATION, N>>>(pos, ru01);
  init_vel<<<POPULATION, N>>>(vel, ru02);
  ffill<<<POPULATION, N>>>(evals, 999999);
  cudaDeviceSynchronize(); clock_t start, end;
  double cpu_time_used; start = clock();


  int i;
  for (i = 0; i < N_ITERS; i ++){

    U_01<<<N, POPULATION>>>(ru01, states);
    U_01<<<N, POPULATION>>>(ru02, states);

    F_KEY<<<POPULATION, N>>>(pos, evals);

    update_min<<<POPULATION, N>>>(pos, min, evals, min_evals);

    arrayReduction<<<1, POPULATION>>>(min_evals);

    update_gMin<<<1, N>>>(min, min_evals, gMin);

    update_vel<<<POPULATION, N>>>(vel, pos, min, gMin, ru01, ru02);

    update_pos<<<POPULATION, N>>>(pos, vel);

  }

  cudaDeviceSynchronize();
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;


  cudaDeviceSynchronize();
  printf("\n");
  printf("[-] About your GPU ...\n");
  int dev = 0;
  cudaSetDevice(dev);

  cudaDeviceProp devProps;
  if (cudaGetDeviceProperties(&devProps, dev) == 0)
  {
      printf("[-]    Model            : %s \n[-]    Mem. global      : %d Bytes (%d MB) \n[-]    Compute cap.     : v%d.%d \n[-]    Clock speed      : %d MHz\n",
             devProps.name, (int)devProps.totalGlobalMem,
             (int)devProps.totalGlobalMem / 1000000,
             (int)devProps.major, (int)devProps.minor,
             (int)devProps.clockRate/1000);
  }

  printf("[-] About the experiment ...\n");
  printf("[-]    Function         : %s\n", F_NAME);
  printf("[-]    Heuristic method : PSO\n");
  printf("[-]    Variables        : %d\n", N);
  printf("[-]    Population       : %d\n", POPULATION);
  printf("[-]    Bounds           : [%d, %d]\n", LB, UB);
  printf("[-]    Iters            : %d\n", N_ITERS);
  printf("[-] About the results ...\n");
  printf("[-]    Time required    : %f\n", cpu_time_used);
  printf("[-]    Minimum location : ");


  arrayReduction<<<1, POPULATION>>>(evals);
  print_iMin<<<1,1>>>(pos);

  cudaDeviceSynchronize();
  printf("\n");
  printf("\n");
  return 0;
}
