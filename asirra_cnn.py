from functions_used import load_cifar_10, define_model_cifar10, train_and_evaluate_model, summarize_diagnostics
from dataset.asirra import read_asirra_subset

if __name__ == '__main__':

    # Parameter Setting
    learning_rate = 0.001
    training_epoch = 30
    batch_size = 64
    display_step = 20

    train_images, train_labels, test_images, test_labels = load_cifar_10()
    # change needed!
    # print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

    model = define_model_cifar10(learning_rate)
    history = train_and_evaluate_model(model, train_images, train_labels,
                                       test_images, test_labels, batch_size, training_epoch)
    summarize_diagnostics(history)
