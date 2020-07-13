#include <iostream>
#include "osqp.h"
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




int countNonZero(MatrixXd matrix, int dataset_samples_count){
    int count = 0;
    for(int i = 0; i<dataset_samples_count;i++){
        for(int j = 0; j<dataset_samples_count;j++) {
            if(matrix(i,j) != 0.0){
                count++;
            }
        }
    }
    return count;

}




double *buildP_x(MatrixXd matrix, int dataset_samples_count){
    int length = countNonZero(matrix,dataset_samples_count);
    double *P_x = new double[length];
    int k = 0;

    for(int i = 0; i<dataset_samples_count;i++){
        for(int j = 0; j<dataset_samples_count;j++) {
            if(matrix(i,j) != 0.0){
                P_x[k] = matrix(i,j);
                k++;
            }
        }
    }


    return P_x;
}




double *buildP_i(MatrixXd matrix, int dataset_samples_count) {
    int length = countNonZero(matrix, dataset_samples_count);
    MatrixXd matrix2 = matrix.transpose();
    double *P_i = new double[length];
    int k = 0;
    for (int i = 0; i < dataset_samples_count; i++) {
        int l = 0;
        for (int j = 0; j < dataset_samples_count; j++) {
            if(matrix2(i,j) != 0){
                P_i[k] = l;
                k++;
            }
            l++;
        }
    }
    return P_i;

}



bool *buildBoolP_p(MatrixXd matrix, int dataset_samples_count) {
    int length = countNonZero(matrix, dataset_samples_count);
    bool *BP_p = new bool[length];
    MatrixXd matrix2 = matrix.transpose();
    int k = 0;

    for(int i = 0; i<dataset_samples_count;i++){
        double *tab;
        tab = matrixRowToTab(matrix2,dataset_samples_count,dataset_samples_count,i);
        int justOne = 0;
        for(int j = 0; j<dataset_samples_count;j++){

            if(tab[j] != 0 && justOne<1){
                if(firstNonNulInTabColumn(tab,dataset_samples_count,tab[j])){
                    justOne++;
                }
                BP_p[k] = firstNonNulInTabColumn(tab,dataset_samples_count,tab[j]);
                k++;
            }else{ // !!! if(tab[j] != 0){
                BP_p[k] = false;
                k++;
            }

        }
    }
    return BP_p;
}




double *buildP_p(MatrixXd matrix, int dataset_samples_count) {
    double *P_p = new double[dataset_samples_count+1];
    int k = 0;

    int lengthTab = dataset_samples_count*dataset_samples_count;
    bool *tab = buildBoolP_p(matrix,dataset_samples_count);

    int lengthP_i = countNonZero(matrix, dataset_samples_count);

    for(int i = 0; i<lengthTab;i++){
        if(tab[i]){
            P_p[k] = k;
            k++;
        }
    }
    //dernière case
    P_p[dataset_samples_count] = lengthP_i;

    return P_p;

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




    MatrixXd m = buildBigMatrix(X,Y,nbreEnter,nbreFeature);


    MatrixXd tm = getUpperTriangularPart(m,nbreEnter);


    double *tab = buildP_x(tm,nbreEnter);
    double *tab2 = buildP_i(tm,nbreEnter);
    bool *tab3 = buildBoolP_p(tm,nbreEnter);
    double *tab4 = buildP_p(tm,nbreEnter);

    
/*
//REFERENCE
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

    int lengtNonZero = countNonZero(tm2,p1);

    double *table = buildP_x(tm2,p1);
    std::cout << "P_x" << std::endl;
    printTab(table,lengtNonZero);

    double *table2 = buildP_i(tm2,p1);
    std::cout << "P_i" << std::endl;
    printTab(table2,lengtNonZero);

    double *table3 = buildP_p(tm2,p1);
    std::cout << "P_p" << std::endl;
    printTab(table3,p1+1);


*/




// Workspace structures
    OSQPWorkspace *work;


    return 0;
}
