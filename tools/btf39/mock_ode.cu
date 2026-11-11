// Mock model: 9 states, NY=10 (one algebraic). Blocks {0,1,2} (stiff, cubic
// saturation), {3,4} driven by block 1, {5..8} decoupled; entry (8,0) appears
// only when |x0| > 0.3 (tests pattern growth).
#define MOCK_Y_LEN 10
#define MOCK_P_LEN 1
#define MOCK_STATE_COUNT 9
#define MOCK_DELAY_HISTORY_STRIDE 1
__global__ void Mock_load_starts_batch(double*y,double*p,int,int,int){for(int i=0;i<10;i++)y[i]=0; p[0]=1;}
__global__ void Mock_initialize_batch(double,double*,double*,int*st,int,int,int){*st=0;}
__global__ void Mock_delay_init_batch(double,double*,double*,double*,int,int,int,int){}
__global__ void Mock_delay_apply_batch(double,double*,double*,double*,int,int,int,int){}
__global__ void Mock_delay_commit_batch(double,double*,double*,double*,int,int,int,int){}
__global__ void Mock_event_update_batch(double,double*,double*,int*n,int,int,int){*n=0;}
__global__ void Mock_project_manifold_batch(double,double*,double*,int*,int,int,int){}
__global__ void Mock_rhs_batch(double t,double*y,double*p,double*d,int*st,int,int,int,int){
  *st=0; double s=sin(2*M_PI*60*t);
  y[9]=y[0]+y[1];
  d[0]=-1e6*(y[0]-y[1])+1e6*s - 1e4*y[0]*y[0]*y[0];
  d[1]=1e5*(y[0]-y[1])-2e5*y[1]+3e4*y[2];
  d[2]=-5e5*y[2]+1e5*y[9];
  d[3]=-1e5*y[3]+2e4*y[4]+5e4*y[1];
  d[4]=3e4*y[3]-1e5*y[4];
  d[5]=-2e6*y[5]+1e6*y[6]+500*s;
  d[6]=1e6*y[5]-2e6*y[6]+1e6*y[7];
  d[7]=1e6*y[6]-3e6*y[7]+2e5*y[8];
  d[8]=-1e5*y[8]+5e4*y[7]+(fabs(y[0])>0.3? 1e4*y[0]:0.0);
}
