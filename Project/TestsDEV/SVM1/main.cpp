#include <iostream>
#include "osqp.h"
#include "cs.h"
#include <Eigen/Dense>
using Eigen::MatrixXd;


//juste for test in C++
double *getPartsOfTab(int start, int stop, double *tab) {
    //start and stop are in newTab (inclus)
    auto newTab = new double[(stop - start) + 1];

    int j = 0;
    for (int i = start; i <= stop; i++) {
        newTab[j] = tab[i];
        j++;
    }

    return newTab;

}



void printTab(double *tab, int tablLength){
    for(int i = 0; i<tablLength;i++){
        printf("%f ",tab[i]);
    }
    printf("\n");
}




void printTabInt(int *tab, int tablLength){
    for(int i = 0; i<tablLength;i++){
        printf("%d ",tab[i]);
    }
    printf("\n");
}





MatrixXd tabToMatrix( double *tab, int tabSize, int exampleCount_or_ROW, int inputsSize_or_COL) {

    MatrixXd m(exampleCount_or_ROW, inputsSize_or_COL);
    int i = 0;
    int j = 0;
    while (i < tabSize) {

        for (int k = 0; k < inputsSize_or_COL - 1; k++) {
            m(j, k) = tab[i];
            i++;
        }
        m(j, inputsSize_or_COL - 1) = tab[i];
        i++;
        j++;


    }


    return m;

}




bool firstNonNulInTabColumn(double *tab,int tabLength, double val){
    if(val == 0){
        return false;
    }
    for(int i = 0; i<tabLength;i++){
        if(tab[i] == val){
            if(i == 0){

                return true;
            }else{

                int j = i-1;
                while (j >= 0) {
                    if (tab[j] != 0) {
                        return false;
                    }
                    j--;
                }
                return true;

            }
        }
    }
    return true;
}





double *matrixRowToTab(MatrixXd matrix, int row, int column, int whichRow) {
    double *tab = new double[column];


    for(int i = 0; i<column;i++){
        tab[i] = matrix(whichRow,i);
    }

    return tab;
}





MatrixXd buildBigMatrix(double *inputs, double *expectedOutputs, int dataset_samples_count, int dataset_sample_features_count){
    MatrixXd matrix(dataset_samples_count,dataset_samples_count);
    //puts the Y
    for(int i = 0;i<dataset_samples_count;i++){
        for(int j  =0; j<dataset_samples_count;j++){
            double tabTemp1[1] = {expectedOutputs[i]};
            MatrixXd temp1 = tabToMatrix(tabTemp1,1,1,1);
            double tabTemp2[1] = {expectedOutputs[j]};
            MatrixXd temp2 = tabToMatrix(tabTemp2,1,1,1);
            MatrixXd result1 = temp1 * temp2;
            auto result2  = result1(0,0);
            matrix(i,j) = result2;
        }
    }

    int k = 0;
    for(int i = 0;i+dataset_sample_features_count<=dataset_samples_count*dataset_sample_features_count;
    i = i+dataset_sample_features_count){
        int l = 0;
        for(int j = 0;j+dataset_sample_features_count<=dataset_samples_count*dataset_sample_features_count;
            j = j+dataset_sample_features_count){
            double *tabTemp3 = getPartsOfTab(i,i+dataset_sample_features_count-1,inputs);
            MatrixXd temp3 = tabToMatrix(tabTemp3,dataset_sample_features_count,dataset_sample_features_count,1);
            MatrixXd temp4 = temp3.transpose();
            double *tabTemp5 = getPartsOfTab(j,j+dataset_sample_features_count-1,inputs);
            MatrixXd temp5 = tabToMatrix(tabTemp5,dataset_sample_features_count,dataset_sample_features_count,1);
            MatrixXd result3 = temp4 * temp5;
            auto result4 = result3(0,0);
            matrix(k,l) = matrix(k,l)*result4;
            l++;
        }
        k++;

    }

    return matrix;
}






MatrixXd getUpperTriangularPart(MatrixXd matrix, int dataset_samples_count){
    for(int i = 0; i<dataset_samples_count;i++){
        for(int j = 0; j<dataset_samples_count;j++){
            if(i>j){
                matrix(i,j) = 0.0;
            }
        }
    }

    return matrix;
}





int countNonZero(MatrixXd matrix, int x, int y){
    int count = 0;
    for(int i = 0; i<x;i++){
        for(int j = 0; j<y;j++) {
            if(matrix(i,j) != 0){
                count++;
            }
        }
    }
    return count;

}





double *buildP_x(MatrixXd matrix, int row, int column){
    int length = countNonZero(matrix,row,column);
    double *P_x = new double[length];
    int k = 0;

    for(int i = 0; i<row;i++){
        for(int j = 0; j<column;j++) {
            if(matrix(i,j) != 0.0){
                P_x[k] = matrix(i,j);
                k++;
            }
        }
    }


    return P_x;
}





int *buildP_i(MatrixXd matrix, int row, int column) {
    int length = countNonZero(matrix, row, column);
    MatrixXd matrix2 = matrix.transpose();
    std::cout << matrix2 << std::endl;
    int *P_i = new int[length];
    int k = 0;
    int m = 0;
    for (int i = 0; i < column; i++) {
        if(row <= 1){
            if (matrix2(i, 0) != 0) {
                P_i[k] = m;
                m++;
                k++;
            }
        }else {
            int l = 0;
            for (int j = 0; j < row; j++) {
                if (matrix2(i, j) != 0) {
                    P_i[k] = l;
                    k++;
                }
                l++;
            }
        }
    }

    return P_i;

}




bool *buildBoolP_p(MatrixXd matrix, int row, int column) {
    int length = countNonZero(matrix, row, column);
    bool *BP_p = new bool[length];
    MatrixXd matrix2 = matrix.transpose();

    int k = 0;

    if(row <= 1){
        double *tab = matrixRowToTab(matrix, row, column, 0);
        int justOne;
        justOne = 0;
        for(int i = 0; i<column;i++){
            if (tab[i] != 0 && justOne < 1) {
                if (firstNonNulInTabColumn(tab, column, tab[i])) {
                    justOne++;
                    BP_p[k] = true;
                    k++;
                }
            } else if (tab[i] != 0) {
                BP_p[k] = false;
                k++;
            }
        }
    }else {
        for (int i = 0; i < column; i++) {
            double *tab;
            tab = matrixRowToTab(matrix2, column, row, i);
            int justOne;
            justOne = 0;
            for (int j = 0; j < row; j++) {
                if (tab[j] != 0 && justOne < 1) {
                    if (firstNonNulInTabColumn(tab, row, tab[j])) {
                        justOne++;
                        //}
                        BP_p[k] = true;
                        k++;
                    }
                } else if (tab[j] != 0) {
                    BP_p[k] = false;
                    k++;
                }

            }

        }
    }

    return BP_p;
}





int *buildP_p(MatrixXd matrix, int row, int column) {
    int *P_p;
    if(row <= 1){
        P_p = new int[row + 1];
    }else {
        P_p = new int[column + 1];
    }
        int k = 0;

        //int lengthTab = row*column;
        bool *tab = buildBoolP_p(matrix, row, column);


        for (int i = 0; i < countNonZero(matrix, row, column); i++) {
            if (tab[i]) {
                P_p[k] = i;
                k++;
            }
        }

     if(row <=1){
        P_p[row] = countNonZero(matrix, row, column) + 1;
     }else {
         //dernière case
         P_p[column] = countNonZero(matrix, row, column) + 1;
     }

    return P_p;

}




double *buildTabOfNumber(int dataset_samples_count, int number){
    double *tab = new double[dataset_samples_count];
    for(int i = 0; i<dataset_samples_count;i++){
        tab[i] = number;
    }
    return tab;
}



int main() {
    std::cout << "Hello, World!" << std::endl;


    const int nbreFeature = 2;
    const int nbreEnter  = 10;

    double X[nbreFeature*nbreEnter] = {
            1.0,3.0,
            1.0,4.0,
            2.0,2.0,
            2.0,5.0,
            3.0,3.0,

            6.0,7.0,
            7.0,8.0,
            8.0,9.0,
            8.0,5.0,
            9.0,8.0,


    };

    double Y[nbreEnter] = {
            -1,
            -1,
            -1,
            -1,
            -1,

            1,
            1,
            1,
            1,
            1
    };




    MatrixXd matrix = buildBigMatrix(X,Y,nbreEnter,nbreFeature);
    MatrixXd transMatrix = getUpperTriangularPart(matrix,nbreEnter);
    MatrixXd Ymat = tabToMatrix(Y,nbreEnter,1,nbreEnter);

    double *P_x = buildP_x(transMatrix,nbreEnter,nbreEnter);
    c_float *P_x2 = P_x;
    int P_nnz = countNonZero(transMatrix,nbreEnter,nbreEnter);
    c_int P_nnz2 = P_nnz;
    int *P_i = buildP_i(transMatrix,nbreEnter,nbreEnter);
    c_int *P_i2 = reinterpret_cast<c_int *>(P_i);
    int *P_p = buildP_p(transMatrix,nbreEnter,nbreEnter);
    c_int *P_p2 = reinterpret_cast<c_int *>(P_p);

    double *q = buildTabOfNumber(nbreEnter,-1.0);
    c_float *q2 = q;

    int row = 1;
    int column = nbreEnter;
    double *A_x = buildP_x(Ymat,row,column);
    c_float *A_x2 = A_x;
    int A_nnz = countNonZero(Ymat,row,column);
    c_int A_nnz2 = A_nnz;
    int *A_i = buildP_i(Ymat,row,column);
    c_int *A_i2 = reinterpret_cast<c_int *>(A_i);
    int *A_p = buildP_p(Ymat,row,column);
    c_int *A_p2 = reinterpret_cast<c_int *>(A_p);

    double *l =  buildTabOfNumber(nbreEnter,0);
    c_float *l2 = l;
    double *u =  buildTabOfNumber(nbreEnter,0);
    c_float *u2 = u;

    int n = 2;
    c_int n2 = n;
    int m = 3;
    c_int m2 = m;

    printf("Debug 1\n");
    // Exitflag
    c_int exitflag = 0;

    // Workspace structures
    OSQPWorkspace *work;
    OSQPSettings  *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
    OSQPData      *data     = (OSQPData *)c_malloc(sizeof(OSQPData));

    printf("Debug 2\n");
    // Populate data
    if (data) {
        data->n = n2;
        data->m = m2;
        data->P = csc_matrix(data->n, data->n, P_nnz2, P_x2, P_i2, P_p2);
        data->q = q2;
        data->A = csc_matrix(data->m, data->n, A_nnz2, A_x2, A_i2, A_p2);
        data->l = l2;
        data->u = u2;
    }

    printf("Debug 3\n");
    // Define solver settings as default
    if (settings) {
        osqp_set_default_settings(settings);
        settings->alpha = 1.0; // Change alpha parameter
    }

    printf("Debug 4\n");
    // Setup workspace
    exitflag = osqp_setup(&work, data, settings);
    c_int ret = exitflag;
    printf("%d\n",ret);

    printf("Debug 5\n");
    // Solve Problem
    osqp_solve(work);

    printf("Debug 6\n");
    /*
    for(int i = 0; i<nbreEnter;i++){
        printf("%f ",work->solution->x[i]);
    }
    printf("%s\n",work->info->status);

    */

    // Cleanup
    if (data) {
        if (data->A) c_free(data->A);
        if (data->P) c_free(data->P);
        c_free(data);
    }
    if (settings) c_free(settings);

    return exitflag;



/*

//REFERENCE 1
    int p1 = 2;
    int p2 = 2;

    MatrixXd m2(2,2);
    m2(0,0) = 4.0;
    m2(0,1) = 1.0;
    m2(1,0) = 1.0;
    m2(1,1) = 2.0;




    std::cout << "matrice" << std::endl;
    std::cout << m2 << std::endl;

    MatrixXd tm2 = getUpperTriangularPart(m2,p1);
    std::cout << "matrice traingulaire" << std::endl;
    std::cout << tm2 << std::endl;


    int lengtNonZero = countNonZero(tm2,p1,p1);
    std::cout << "Non zero" << std::endl;
    printf("%d \n",lengtNonZero);

    double *table = buildP_x(tm2,p1,p1);
    std::cout << "P_x" << std::endl;
    printTab(table,lengtNonZero);

    int *table2 = buildP_i(tm2,p1,p1);
    std::cout << "P_i" << std::endl;
    printTabInt(table2,lengtNonZero);

    int *table3 = buildP_p(tm2,p1,p1);
    std::cout << "P_p" << std::endl;
    printTabInt(table3,p1+1);




//REFERENCE 2
    int p3 = 3;
    int p4 = 2;

    MatrixXd m3(3,2);
    m3(0,0) = 1;
    m3(0,1) = 1;
    m3(1,0) = 1;
    m3(1,1) = 0;
    m3(2,0) = 0;
    m3(2,1) = 8;



    std::cout << "matrice" << std::endl;
    std::cout << m3 << std::endl;



    int lengtNonZero = countNonZero(m3,p3,p4);
    std::cout << "Non zero" << std::endl;
    printf("%d \n",lengtNonZero);

    double *table = buildP_x(m3,p3,p4);
    std::cout << "P_x" << std::endl;
    printTab(table,lengtNonZero);

    int *table2 = buildP_i(m3,p3,p4);
    std::cout << "P_i" << std::endl;
    printTabInt(table2,lengtNonZero);

    int *table3 = buildP_p(m3,p3,p4);
    std::cout << "P_p" << std::endl;
    printTabInt(table3,p4+1);

*/




    return 0;
}
