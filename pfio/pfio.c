/* 
* a test program to read in PFB
*/
#include "Python.h"
#include "endian.h"
#include <arpa/inet.h>
#include <stdint.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

uint64_t pfntohll(uint64_t value);


#define READINT(V,f) {uint32_t buf; \
                         fread(&buf, 4, 1, f);\
                         uint32_t temp =  ntohl(buf);\
                         V = *(int*)&temp;}
#define READDOUBLE(V,f) {uint64_t buf; \
                         fread(&buf, 8, 1, f);\
                         uint64_t temp =  pfntohll(buf);\
                         V = *(double*)&temp;}


/* #### Utility functions ######################### */
typedef unsigned char byte;

int BigEndianSystem;

short (*BigShort) ( short s );
short (*LittleShort) ( short s );
int (*BigLong) ( int i );
int (*LittleLong) ( int i );
double (*BigFloat) ( double f );
double (*LittleFloat) ( double f );


//adapted from Quake 2 source and tools_io.c

void tools_WriteFloat(
                      FILE * file,
                      float *ptr,
                      int    len)
{
  int i;
  float *data;

  union {
//      double number;
//      char buf[8];
    float number;
    char buf[4];
  } a, b;

  /* write out each double with bytes swaped                               */
  for (i = len, data = ptr; i--; )
  {
    a.number = *data++;
    b.buf[0] = a.buf[3];
    b.buf[1] = a.buf[2];
    b.buf[2] = a.buf[1];
    b.buf[3] = a.buf[0];

    fwrite(&b.number, sizeof(float), 1, (FILE*)file);
  }
}

void tools_WriteDouble(
                       FILE *  file,
                       double *ptr,
                       int     len)
{
  int i;
  double *data;

  union {
    double number;
    char buf[8];
  } a, b;

  /* write out each double with bytes swaped                               */
  for (i = len, data = ptr; i--; )
  {
    a.number = *data++;
    b.buf[0] = a.buf[7];
    b.buf[1] = a.buf[6];
    b.buf[2] = a.buf[5];
    b.buf[3] = a.buf[4];
    b.buf[4] = a.buf[3];
    b.buf[5] = a.buf[2];
    b.buf[6] = a.buf[1];
    b.buf[7] = a.buf[0];

    fwrite(&b.number, sizeof(double), 1, (FILE*)file);
  }
}

void tools_WriteInt(
                    FILE * file,
                    int *  ptr,
                    int    len)
{
  int i;
  int *data;

  union {
    long number;
    char buf[4];
  } a, b;


  /* write out int with bytes swaped                                       */
  for (i = len, data = ptr; i--; )
  {
    a.number = *data++;
    b.buf[0] = a.buf[3];
    b.buf[1] = a.buf[2];
    b.buf[2] = a.buf[1];
    b.buf[3] = a.buf[0];

    fwrite(&b.number, sizeof(int), 1, (FILE*)file);
  }
}

short ShortSwap( short s )
{
	byte b1, b2;
	
	b1 = s & 255;
	b2 = (s >> 8) & 255;

	return (b1 << 8) + b2;
}

short ShortNoSwap( short s )
{
	return s;
}

int LongSwap (int i)
{
	byte b1, b2, b3, b4;

	b1 = i & 255;
	b2 = ( i >> 8 ) & 255;
	b3 = ( i>>16 ) & 255;
	b4 = ( i>>24 ) & 255;

	return ((int)b1 << 24) + ((int)b2 << 16) + ((int)b3 << 8) + b4;
}

int LongNoSwap( int i )
{
	return i;
}

double FloatSwap( double f )
{
	union
	{
	    double f;
	    char b[8];
	} dat1, dat2;

	dat1.f = f;
	dat2.b[0] = dat1.b[7];
	dat2.b[1] = dat1.b[6];
	dat2.b[2] = dat1.b[5];
	dat2.b[3] = dat1.b[4];
	dat2.b[4] = dat1.b[3];
	dat2.b[5] = dat1.b[2];
	dat2.b[6] = dat1.b[1];
	dat2.b[7] = dat1.b[0];	
	
	return dat2.f;
}

double FloatNoSwap( double f )
{
	return f;
}

void InitEndian( void )
{
	//clever little trick from Quake 2 to determine the endian
	//of the current system without depending on a preprocessor define

	byte SwapTest[2] = { 1, 0 };
	
	if( *(short *) SwapTest == 1 )
	{
		//little endian
		BigEndianSystem = 1;
               // printf("This is a little endian machine!\n");
		//set func pointers to correct funcs
		BigShort = ShortSwap;
		LittleShort = ShortNoSwap;
		BigLong = LongSwap;
		LittleLong = LongNoSwap;
		BigFloat = FloatSwap;
		LittleFloat = FloatNoSwap;
	}
	else
	{
		//big endian
		BigEndianSystem = 0;
                //printf("This is a big endian machine!\n");

		BigShort = ShortNoSwap;
		LittleShort = ShortSwap;
		BigLong = LongNoSwap;
		LittleLong = LongSwap;
		BigFloat = FloatNoSwap;
		LittleFloat = FloatSwap;
	}
}

/* #### Main read and write pfb files functions ######################### */

static PyObject * pfread(PyObject *self, PyObject *args);
static PyObject * pfwrite(PyObject *self, PyObject *args);
static PyMethodDef pf_funcs[] = { { "pfread", pfread, METH_VARARGS },
                                  { "pfwrite", pfwrite, METH_VARARGS },
                                  { NULL, NULL } };
   
/*******************************************************************************
Function: pfread(char *file_name)
Reads a pfb file into a 3 dimensional PyArraytObject
*******************************************************************************/
static PyObject * pfread(PyObject *self, PyObject *args){
  FILE *fp;
  char *pfbfile;
    
  if (!PyArg_ParseTuple(args, "s", &pfbfile)) {
    return NULL;
  }
   
  PyArrayObject *array_out;
  npy_intp dims[3];
  double *val_array;
   
  // header variables 
  double     X, Y, Z;
  int        NX, NY, NZ, N;
  double     DX, DY, DZ;
  int        num_subgrids;
  // subgrid header variables
  int        x, y, z;
  int        nx, ny, nz;
  int        rx, ry, rz;
  // looping variables
  int        nsg, j, k, i;
  // address calculation
  int        qq;
    
  // open the input file 
  if ((fp = fopen(pfbfile, "rb")) == NULL) {
    perror("Error opening pfbfile");
    return NULL;
  }
  /* read in header information */
  READDOUBLE(X,fp);
  READDOUBLE(Y,fp);
  READDOUBLE(Z,fp);
  READINT(NX,fp); 
  READINT(NY,fp); 
  READINT(NZ,fp); 
  READDOUBLE(DX,fp);
  READDOUBLE(DY,fp);
  READDOUBLE(DZ,fp);
   
  // allocate output array 
  dims[0] = NZ;
  dims[1] = NY;
  dims[2] = NX;
  N = NZ*NY*NX;
    
  array_out = (PyArrayObject*) PyArray_SimpleNew(3, dims, NPY_DOUBLE);
  val_array = (double*)PyArray_DATA(array_out);
    
  READINT(num_subgrids,fp); 
  for (nsg = num_subgrids;nsg>0; nsg--){
    // read subgrid header
    READINT(x,fp); 
    READINT(y,fp); 
    READINT(z,fp); 
    READINT(nx,fp); 
    READINT(ny,fp); 
    READINT(nz,fp); 
    READINT(rx,fp); 
    READINT(ry,fp); 
    READINT(rz,fp); 
     
    // read values for subgrid 
    for (k = 0; k < nz; k++){
      for (j = 0; j <ny; j++){
        for (i = 0; i < nx; i++){   
	  qq = ((z+k)*(NX*NY)) + ((NX)*(y+j)+(x+i));
          READDOUBLE(val_array[qq],fp);
        }
      }
    }
  }
  fclose(fp);
  return PyArray_Return(array_out);
}

static PyObject * pfwrite(PyObject *self, PyObject *args)
{
    FILE *fp;
    
    PyArrayObject *matin;
    
    char *pfbfile;
    
    double *val_array;
    double *new_val_array;
    PyArrayObject *array_out;
    
    int        j, k, i, iteri;
    int 		num_iter;
    npy_intp dims[3];
	
    double     X, Y, Z;
    int        NX, NY, NZ;
    double     DX, DY, DZ;
    
    if (!PyArg_ParseTuple(args, "O!sdddddd", &PyArray_Type, &matin, &pfbfile,
    										&X, &Y, &Z,
    										&DX, &DY, &DZ)) {
      return NULL;
   }
    
    NZ = PyArray_DIMS(matin)[0];
    NY = PyArray_DIMS(matin)[1];
    NX = PyArray_DIMS(matin)[2];
    
	val_array = (double*)PyArray_DATA(matin);
    
    fp = fopen(pfbfile, "w");
    
    int ns = 1;
    int x = 0;
	int y = 0;
	int z = 0;
	int nx = NX;
	int ny = NY;
	int nz = NZ;
	int rx = 1;
	int ry = 1;
	int rz = 1;
    
    tools_WriteDouble(fp, &X, 1);
	tools_WriteDouble(fp, &Y, 1);
	tools_WriteDouble(fp, &Z, 1);
	tools_WriteInt(fp, &NX, 1);
	tools_WriteInt(fp, &NY, 1);
	tools_WriteInt(fp, &NZ, 1);
	tools_WriteDouble(fp, &DX, 1);
	tools_WriteDouble(fp, &DY, 1);
	tools_WriteDouble(fp, &DZ, 1);
	
	tools_WriteInt(fp, &ns, 1);

	tools_WriteInt(fp, &x, 1);
	tools_WriteInt(fp, &y, 1);
	tools_WriteInt(fp, &z, 1);
	tools_WriteInt(fp, &nx, 1);
	tools_WriteInt(fp, &ny, 1);
	tools_WriteInt(fp, &nz, 1);
	tools_WriteInt(fp, &rx, 1);
	tools_WriteInt(fp, &ry, 1);
	tools_WriteInt(fp, &rz, 1);
	
	//Trying to reverse back the array
	
	if (NZ < 3) {
		dims[0] = NZ;
		dims[1] = NY;
		dims[2] = NX;
	
		array_out = (PyArrayObject*) PyArray_SimpleNew(3, dims, NPY_DOUBLE);
		new_val_array = (double*)PyArray_DATA(array_out);
		for (k = 0; k < NZ; k++)
		{
			for (j = 0; j < (NY/2 +1) ; j++)
			{
				for (i = 0; i < NX; i++)
				{
					//double temp = val_array[k*NX*NY+j*NX+i];
					//val_array[k*NX*NY+j*NX+i]=val_array[k*NX*NY+(NY-j-1)*NX+i];
					//val_array[k*NX*NY+(NY-j-1)*NX+i] = temp; 
					new_val_array[k*NX*NY+j*NX+i]=val_array[k*NX*NY+(NY-j-1)*NX+i];
					new_val_array[k*NX*NY+(NY-j-1)*NX+i]=val_array[k*NX*NY+j*NX+i];
				}
			}
		}
	
		tools_WriteDouble(fp, new_val_array, nx*ny*nz);
	}
	else {
		num_iter = NZ/3 +1 ;
		//printf("num_iter = %i \n", num_iter);
		for (iteri = 0; iteri < num_iter; iteri++) {
			int starti = iteri*3;
			int endi = (iteri+1)*3;
			int mini_nz = 3;
			if (endi > NZ) {
				endi = NZ;
				mini_nz = endi - starti;
			}
			//printf("starti = %i \n endi = %i \n mini_nz = %i \n", starti, endi, mini_nz);
			dims[0] = mini_nz;
			dims[1] = NY;
			dims[2] = NX;
	
			array_out = (PyArrayObject*) PyArray_SimpleNew(3, dims, NPY_DOUBLE);
			new_val_array = (double*)PyArray_DATA(array_out);
			for (k = starti; k < endi; k++)
			{
				for (j = 0; j < (NY/2 +1) ; j++)
				{
					for (i = 0; i < NX; i++)
					{
						//double temp = val_array[k*NX*NY+j*NX+i];
						//val_array[k*NX*NY+j*NX+i]=val_array[k*NX*NY+(NY-j-1)*NX+i];
						//val_array[k*NX*NY+(NY-j-1)*NX+i] = temp; 
						new_val_array[(k-iteri*3)*NX*NY+j*NX+i]=val_array[k*NX*NY+(NY-j-1)*NX+i];
						new_val_array[(k-iteri*3)*NX*NY+(NY-j-1)*NX+i]=val_array[k*NX*NY+j*NX+i];
					}
				}
			}
	
			tools_WriteDouble(fp, new_val_array, nx*ny*mini_nz);
		}
	}
	
	Py_RETURN_NONE;
}

void freeme(double *aptr)
{
    printf("freeing address: %p\n", aptr);
    free(aptr);
}


static struct PyModuleDef pfio =
{
    PyModuleDef_HEAD_INIT,
    "pfio", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    pf_funcs
};

PyMODINIT_FUNC PyInit_pfio(void)
{
    PyObject* po = PyModule_Create(&pfio);
  	import_array();
    return po;
}

uint64_t pfntohll(uint64_t value) {
    if (htonl(1) != 1){
        const uint32_t high_part = htonl((uint32_t)(value >> 32));
        const uint32_t low_part = htonl((uint32_t)(value));// & 0xFFFFFFFFLL));
        uint64_t retval = (uint64_t)low_part << 32;
        retval = retval | high_part;
        return retval;
    } 
    return value;
}
