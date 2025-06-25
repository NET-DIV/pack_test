#include <stdio.h>
#include <math.h>


#define DT 0.01
#define SIM_TIME 100.0
#define STEPS (int)(SIM_TIME / DT)
#define LEARNING_RATE 0.01
#define EPOCHS 100

#define E_K -77.0  
#define V_M -65.0  
double target_function(double t) {
    return sin(t / 10.0);
}

double alpha_n(double Vm) {
    return (10.0 - Vm) / (100.0 * (exp((10.0 - Vm) / 10.0) - 1.0));
}

double beta_n(double Vm) {
    return 0.125 * exp(-Vm / 80.0);
}

double update_n(double n, double Vm) {
    double a_n = alpha_n(Vm);
    double b_n = beta_n(Vm);
    double dn_dt = a_n * (1.0 - n) - b_n * n;
    return n + DT * dn_dt;
}

double compute_IK(double gK, double n, double Vm) {
    return gK * pow(n, 4) * (Vm - E_K);
}
double dIK_dgK(double n, double Vm) {
    return pow(n, 4) * (Vm - E_K);
}

int main() {
    double gK = 20.0;  
    printf("Training to fit I_K(t) to target(t)...\n");

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double loss = 0.0;
        double grad = 0.0;
        double n = 0.3177; 
        double t = 0.0;

        for (int i = 0; i < STEPS; i++) {
            n = update_n(n, V_M);
            double IK = compute_IK(gK, n, V_M);
            double target = target_function(t);
            double error = IK - target;

            loss += error * error;
            grad += 2 * error * dIK_dgK(n, V_M); 

            t += DT;
        }

        loss /= STEPS;
        grad /= STEPS;

        gK -= LEARNING_RATE * grad;

        printf("Epoch %d: Loss = %.6f, G_K = %.4f\n", epoch + 1, loss, gK);
    }

    return 0;
}