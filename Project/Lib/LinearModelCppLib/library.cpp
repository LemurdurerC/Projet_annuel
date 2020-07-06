#if _WIN32
#define DLLEXPORT _declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include "library.h"
#include <iostream>
#include <cstdlib>
#include <random>
#include <Eigen/Dense>
using Eigen::MatrixXd;


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



void matriceToTab(MatrixXd matrix,  double *tab, int tabSize, int exampleCount_or_ROW, int inputsSize_or_COL) {

    int i = 0;
    int j = 0;
    while (i < tabSize) {

        for (int k = 0; k < inputsSize_or_COL - 1; k++) {
            tab[i] = matrix(j, k);
            i++;
        }
        tab[i] = matrix(j, inputsSize_or_COL - 1);
        i++;
        j++;
    }
}


extern  "C"{


    //feature = parameter (for each example)

    // initialisez les W au hasard entre -1 et 1
    DLLEXPORT double *linear_create_model(int nb_features) {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        auto w = new double[nb_features + 1];
        for (auto i = 0; i < nb_features + 1; i++) {
            w[i] = dist(mt);
        }
        return w;
    }




    DLLEXPORT double linear_predict_model_regression(const double *model, const double *inputs, int inputs_size) {
        auto sum = model[0];
        for (auto i = 0; i < inputs_size; i++) {
            sum += model[i + 1] * inputs[i];
        }
        return sum;
    }



    DLLEXPORT double linear_predict_model_classification(const double *model, const double *inputs, int inputs_size) {
        auto sum = linear_predict_model_regression(model, inputs, inputs_size);
        auto return_val =  sum >= 0 ? 1.0 : -1.0;
        return return_val;
    }



    DLLEXPORT void linear_dispose_model(const double *model) {
        delete[] model;
    }



    DLLEXPORT void linear_train_model_classification(
            double *model,
            const double *dataset_inputs,
            const double *dataset_expected_outputs,
            int dataset_samples_count, //number of all example
            int dataset_sample_features_count, // number of parameter of each example
            double alpha, // learning rate
            int iteration_count
    ) {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> dist(0, dataset_samples_count - 1);

        for (auto it = 0; it < iteration_count; it++) {
            auto k = dist(mt);
            auto inputs_k = dataset_inputs + k * dataset_sample_features_count;

            auto expected_output_k = dataset_expected_outputs[k];

            auto predicted_output_k = linear_predict_model_classification(model, inputs_k, dataset_sample_features_count);

            auto semi_grad = alpha * (expected_output_k - predicted_output_k);
            for (auto i = 0; i < dataset_sample_features_count; i++) {
                model[i + 1] += semi_grad * inputs_k[i];
            }
            model[0] += semi_grad * 1.0;
        }

    }




    DLLEXPORT void linear_train_model_regression(double *model,
                                                 const double *dataset_inputs,
                                                 const double *dataset_expected_outputs,
                                                 int dataset_samples_count, //number of all example
                                                 int dataset_sample_features_count, // number of parameter of each example
                                                 int exepected_output_count
                                                ) {
    // TODO : Pseudo Inverse de Moore-Penrose
    // UTILISEZ EIGEN POUR L INVERSION DE LA MATRICE !!!

    //copy in another tab
    double *X_tab = new double[dataset_samples_count * dataset_sample_features_count];
    double *Y_tab = new double[exepected_output_count];

    for(int i = 0; i<dataset_samples_count * dataset_sample_features_count;i++){
        X_tab[i] = dataset_inputs[i];
    }
    for(int i = 0; i<exepected_output_count;i++){
        Y_tab[i] = dataset_expected_outputs[i];
    }


    //ajouter les entrés fictives pour chaque example pour le X
    MatrixXd X2(dataset_samples_count,1);
    for(int i = 0; i<dataset_samples_count;i++){
        X2(i,0) = 1;
    }

    MatrixXd X1 = tabToMatrix(X_tab, dataset_samples_count * dataset_sample_features_count, dataset_samples_count, dataset_sample_features_count);

    MatrixXd X_mat(X1.rows(), X1.cols()+X2.cols());
    X_mat << X2,X1;

    MatrixXd Y_mat = tabToMatrix(Y_tab, exepected_output_count * 1, exepected_output_count, 1);


    MatrixXd transX = X_mat.transpose();

    MatrixXd a = transX * X_mat;

    MatrixXd b = a.inverse();

    MatrixXd c = b * transX;

    //matrice (3*1)
    MatrixXd W_mat = c * Y_mat;

    matriceToTab(W_mat, model, dataset_sample_features_count+1, dataset_sample_features_count+1, 1);

}




}