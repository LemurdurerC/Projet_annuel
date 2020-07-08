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

    // Workspace structures
    OSQPWorkspace *work;
    
   // std::cout << m << std::endl;


    return 0;
}
