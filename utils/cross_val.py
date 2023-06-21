import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


def cross_validate(
    classifier,
    X_train_orig,
    y_train_orig,
    X_train_mod=None,
    y_train_mod=None,
    num_mod_samples_orig=None,
    splits=5,
):
    skf = StratifiedKFold(n_splits=splits)
    num_orig_samples = X_train_orig.shape[0]
    indices = np.arange(num_orig_samples)
    train_f1_scores = []
    test_f1_scores = []
    for i, (train_index, test_index) in enumerate(skf.split(indices, y_train_orig)):
        print(f"Fold {i}:")
        # select train samples from modified and original data if possible
        X_train_fold = X_train_orig[train_index]
        y_train_fold = y_train_orig[train_index]

        if (
            X_train_mod is not None
            and y_train_mod is not None
            and num_mod_samples_orig is not None
        ):
            # construct selection mask
            num_mod_samples = X_train_mod.shape[0]
            assert num_mod_samples % num_mod_samples_orig == 0
            mod_indices = np.concatenate(
                [
                    train_index + i * num_mod_samples_orig
                    for i in range(num_mod_samples // num_mod_samples_orig)
                ]
            )
            X_train_mod_fold = X_train_mod[mod_indices]
            y_train_mod_fold = y_train_mod[mod_indices]
            X_train_fold = np.concatenate([X_train_fold, X_train_mod_fold])
            y_train_fold = np.concatenate([y_train_fold, y_train_mod_fold])
        print(X_train_fold.shape, y_train_fold.shape)

        X_test_fold = X_train_orig[test_index]
        y_test_fold = y_train_orig[test_index]

        # fit the classifier
        classifier.fit(X_train_fold, y_train_fold)
        y_pred_fold = classifier.predict(X_test_fold)
        f1_micro_test = f1_score(y_test_fold, y_pred_fold, average="micro")
        f1_micro_train = f1_score(
            y_train_fold, classifier.predict(X_train_fold), average="micro"
        )
        print(f"Train F1 micro: {f1_micro_train}")
        print(f"Test F1 micro: {f1_micro_test}")
        train_f1_scores.append(f1_micro_train)
        test_f1_scores.append(f1_micro_test)
    print(f"Average train F1 micro: {np.mean(train_f1_scores)}")
    print(f"Average test F1 micro: {np.mean(test_f1_scores)}")
    # return best classifier
    return classifier
