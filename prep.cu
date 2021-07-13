#pragma once
#include <iostream>
#include <iomanip>

//Matrix füllen
void fillMatrix(DTYPE *A, int size){
  int index = 0;
  for (int i=0;i<size;i++){
    for(int j = 0; j < size; j++){
      index = i*size +j;
      A[index] = (DTYPE)(i);
      //A[index] = (DTYPE)(1);
    }
  }
}
//############################################################################################################

//Vektor füllen
void fillVector(DTYPE *arr, int size) {
  for (int i = 0; i < size; i++)
  //arr[i] = (DTYPE)(i + 1);
  arr[i] = (DTYPE)(i+10);

}
//############################################################################################################

// Berechnung zum Vergleich
void hostAx(DTYPE *a, DTYPE *x, DTYPE *y, int size) {
  DTYPE b = 0.0;
  for (int i = 0; i < size; i++) {
    b = 0.0;
    for (int j = 0; j < size; j++) {
      b += a[j + i * size] * x[j];
    }
    y[i] = b;
  }
}
//############################################################################################################



// Compare results 
int correctness(DTYPE *a, DTYPE *yh, DTYPE *yd, DTYPE *x, int size) {
  float error = 0.0;

  for (int j = 0; j < size; j++) {
    //if (j<10) printf("%f %f\n", yh[j], yd[j]);
    error = max(error, abs(1 - (yh[j] / yd[j])));
    //Abbruchbedingung falls Fehler zu groß wird
    if (abs(1 - (yh[j] / yd[j])) > 1e-3) {
      printf("Fehler in Zeile %i (%f!=%f)\n", j, yh[j],yd[j]);
      return(-1);
    }
  }

  printf("Test bestanden!\n");
  printf("Maximaler Fehler: %f\n",error);
  return 0;
}
//############################################################################################################


