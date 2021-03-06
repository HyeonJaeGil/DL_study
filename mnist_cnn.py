from functions_used import load_mnist, define_alexnet, define_model_mnist, train_and_evaluate_model, summarize_diagnostics


if __name__ == '__main__':

    x_train, y_train, x_test, y_test = load_mnist()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


    # hyper parameters
    learning_rate = 0.001
    training_epochs = 12
    batch_size = 128

    model = define_model_mnist(learning_rate)
    history = train_and_evaluate_model(model, x_train, y_train, x_test, y_test, batch_size, training_epochs)
    summarize_diagnostics(history)
