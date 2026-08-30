import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import warnings

from razdel import tokenize
import pymorphy3
import re
import ast

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, hinge_loss, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import ParameterGrid
from time import time

from sklearn.metrics import classification_report, confusion_matrix


def tokenize_df(df):
    def lemmatize_text(text):
        tokens = [token.text for token in tokenize(text)]
        lemmas = []
        for token in tokens:
            if token.isalpha():
                parsed = morph.parse(token)[0]  # берем первый вариант разбора
                lemmas.append(parsed.normal_form)
        return " ".join(lemmas)
    
    df['tokens'] = df['title'].apply(lambda x: " ".join([token.text for token in tokenize(x)])) 
    morph = pymorphy3.MorphAnalyzer()   # эта штука может убирать названия фильмов и тд
    df['lemmatized'] = df['tokens'].apply(lemmatize_text)
    df = df.drop(columns=["tokens"])


def make_encoders():
    encoder = TfidfVectorizer(lowercase=True,
                            ngram_range=(1, 2),
                            min_df=2,
                            max_df=0.8, 
                            sublinear_tf=True)
    encoder_wb = TfidfVectorizer(analyzer="char_wb",
                            ngram_range=(2, 5), 
                            min_df=3, 
                            sublinear_tf=True, 
                            max_features=300_000)
    encoder_url = TfidfVectorizer(analyzer="char_wb", 
                                ngram_range=(2, 5), 
                                min_df=2, 
                                sublinear_tf=True)
    return [encoder, encoder_wb, encoder_url]


def tuning_sgd(X_train, y_train, X_test, y_test):
    version = int(input("Input the version: "))
    n_samples = X_train.shape[0]
    log_filename = f"log_reg_train_fit_{version}_SGD.log"

    os.makedirs(f"plots_fit_{version}", exist_ok=True)

    logger = logging.getLogger(f"sgd_training_fit_{version}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(file_handler)

    param_grid = {
        "loss": ["hinge", "log_loss"], 
        "penalty" : ["l1", "l2"],
        "alpha": [1e-5, 1e-6],
        "learning_rate": ["optimal"],
        "eta0": [1e-3, 1e-4], 
        "n_iter_no_change": [5, 10], 
        "tol": [1e-4, 1e-5],
        "max_iter": [1000, 2000],
        "validation_fraction": [0.1, 0.2],
        "class_weight": [dict(zip(np.unique(y_train), y_train.shape[0] / (2 * np.bincount(y_train)))), None]
    }

    model_num = 0
    for params in ParameterGrid(param_grid):
        time_start = time()
        model = SGDClassifier(**params,
                            random_state = 42,
                            early_stopping = True)

        print(f"Model num: {model_num}")
        logger.info(
            f"=== New model #{model_num} "
            f"params={params} ==="
        )

        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        if params["loss"] == "log_loss": 
            cur_loss_train = log_loss(y_train, model.predict_proba(X_train))
            cur_loss_test = log_loss(y_test, model.predict_proba(X_test))
        else:
            cur_loss_train = hinge_loss(y_train, model.decision_function(X_train))
            cur_loss_test = hinge_loss(y_test, model.decision_function(X_test))
        logger.info(
            f"Model {model_num}| "
            f"f1={f1_score(y_test, y_pred_test):.4f} recall={recall_score(y_test, y_pred_test):.4f} loss_train={cur_loss_train:.4f} "
            f"loss_test={cur_loss_test:.4f}"
        )

        logger.info(f"Model {model_num} | training finished | total_elapsed={(time() - time_start):.2f}s")
        model_num+=1


def tuning_sgd_partial(X_train, y_train, X_test, y_test):
    version = int(input("Input the version: "))
    n_samples = X_train.shape[0]

    log_file_name = f"log_reg_train_part_fit_{version}_SGD.log"

    os.makedirs("plots_part_fit", exist_ok=True)

    logger = logging.getLogger(f"sgd_part_training_{version}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_name, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(file_handler)


    batch_size_list = [128] 
    n_iter_no_change_list = [5] 
    tol_list = [1e-3, 1e-4]
    max_epochs = 1000
    param_grid = {
        "loss": ["hinge", "log_loss"], 
        "penalty" : ["l1", "l2"],
        "alpha": [1e-6],
        "learning_rate": ["optimal"],
        "eta0": [1e-3, 1e-4], 
        "class_weight": [None, dict(zip(np.unique(y_train), y_train.shape[0] / (2 * np.bincount(y_train))))]
    }
    model_num = 0
    for bs in batch_size_list:
        for n_iter_no_change in n_iter_no_change_list:
            for tol in tol_list:
                for params in ParameterGrid(param_grid):
                    iters_no_change = 0
                    loss_history_train, loss_history_test, f1_history, recall_history = [], [], [], []
                    model = SGDClassifier(
                        **params,
                        random_state = 42,
                        warm_start = True
                        )

                    print(f"Model num: {model_num}")
                    logger.info(
                        f"=== New model #{model_num} | batch_size={bs}, "
                        f"n_iter_no_change={n_iter_no_change}, tol={tol}, params={params} ==="
                    )
                    time_start = time()

                    STOP = False
                    for epoch in range(max_epochs):
                        if STOP:
                            break

                        for idx in range(0, n_samples, bs):
                            start, stop = idx, min(idx+bs, n_samples)
                            X_batch = X_train[start:stop]
                            y_batch = y_train[start:stop]
                            if idx == 0 and epoch == 0:
                                model.partial_fit(X_batch, y_batch, np.unique(y_train))
                            else:
                                model.partial_fit(X_batch, y_batch)

                            if idx%(bs*1000) == 0:
                                y_pred_test = model.predict(X_test)
                                if params["loss"] == "log_loss": 
                                    cur_loss_train = log_loss(y_train, model.predict_proba(X_train))
                                    cur_loss_test = log_loss(y_test, model.predict_proba(X_test))
                                else:
                                    cur_loss_train = hinge_loss(y_train, model.decision_function(X_train))
                                    cur_loss_test = hinge_loss(y_test, model.decision_function(X_test))
                                loss_history_train.append(cur_loss_train)
                                loss_history_test.append(cur_loss_test)
                                f1_history.append(f1_score(y_test, y_pred_test))
                                recall_history.append(recall_score(y_test, y_pred_test))
                                logger.info(
                                    f"Model {model_num} | epoch={epoch} idx={idx} | "
                                    f"f1={f1_history[-1]:.4f} recall={recall_history[-1]:.4f} loss_train={cur_loss_train:.4f} "
                                    f"loss_test={cur_loss_test:.4f}"
                                )
                                # проверка на остановку
                                if iters_no_change >= n_iter_no_change:
                                    STOP = True
                                    logger.info(f"Model {model_num} | early stopping at epoch={epoch} idx={idx}")
                                    break
                                else:
                                    if len(loss_history_train) >= 2 and tol > (loss_history_train[-2] - loss_history_train[-1]):
                                        iters_no_change += 1
                                    else: 
                                        iters_no_change = 0

                        epoch_elapsed = time() - time_start
                        print(f"Эпоха №{epoch}: {epoch_elapsed}")
                        logger.info(f"Model {model_num} | epoch={epoch} finished | elapsed={epoch_elapsed:.2f}s")

                    total_elapsed = time() - time_start
                    print(f"Time elapsed for model {model_num}: {total_elapsed}")
                    logger.info(f"Model {model_num} | training finished | total_elapsed={total_elapsed:.2f}s")

                    fig, ax = plt.subplots()
                    ax.plot(loss_history_train, label="train")
                    ax.plot(loss_history_test, label="test")
                    ax.set_xlabel("checkpoint")
                    ax.set_ylabel("loss")
                    ax.set_title(f"Model {model_num} loss (train vs test)")
                    ax.legend()
                    fig.savefig(f"plots_part_fit/model_{model_num}_loss.png")
                    plt.close(fig)

                    fig, ax = plt.subplots()
                    ax.plot(f1_history, label="f1")
                    ax.plot(recall_history, label="recall")
                    ax.set_xlabel("checkpoint")
                    ax.set_ylabel("score")
                    ax.set_title(f"Model {model_num} f1 / recall")
                    ax.legend()
                    fig.savefig(f"plots_part_fit/model_{model_num}_f1.png")
                    plt.close(fig)

                    model_num+=1


def tuning_logistic_regression(X_train, y_train, X_test, y_test):
    version = int(input("Input the version: "))
    n_samples = X_train.shape[0]
    log_filename = f"log_reg_train_fit_loggin_{version}_LR.log"

    os.makedirs(f"plots_fit_{version}", exist_ok=True)

    logger = logging.getLogger(f"LR_training_fit_{version}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(file_handler)

    param_grid = {
                "l1_ratio": [0.0 ,1.0],
                "C": [1e+2, 10.0, 1.0, 2],
                "tol": [1e-5, 1e-4],
                "solver": ["lbfgs", "liblinear"],
                "max_iter": [100, 120],
                "class_weight": [dict(zip(np.unique(y_train), y_train.shape[0] / (2 * np.bincount(y_train)))), None]
    }

    model_num = 0
    for params in ParameterGrid(param_grid):
        time_start = time()

        print(f"Model num: {model_num}")
        logger.info(
            f"=== New model #{model_num} "
            f"params={params} ==="
        )

        try:
            model = LogisticRegression(**params,
                            random_state = 42)
            model.fit(X_train, y_train)
        except ValueError as e:
            print(f"Выбранная модель не поддерживает такой набор парметров: {params}")
            continue

        y_pred_test = model.predict(X_test)
        cur_loss_train = log_loss(y_train, model.predict_proba(X_train))
        cur_loss_test = log_loss(y_test, model.predict_proba(X_test))
        logger.info(
            f"Model {model_num}| "
            f"f1={f1_score(y_test, y_pred_test):.4f} recall={recall_score(y_test, y_pred_test):.4f} loss_train={cur_loss_train:.4f} "
            f"loss_test={cur_loss_test:.4f}"
        )

        logger.info(f"Model {model_num} | training finished | total_elapsed={(time() - time_start):.2f}s")
        model_num+=1


def tuning_linear_svc(X_train, y_train, X_test, y_test):
    version = int(input("Input the version: "))
    n_samples = X_train.shape[0]
    log_filename = f"log_reg_train_fit_loggin_{version}_LSVC.log"

    os.makedirs(f"plots_fit_LSVC_{version}", exist_ok=True)

    logger = logging.getLogger(f"LSVC_training_fit_{version}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(file_handler)

    param_grid = {
        "penalty": ["l1", "l2"],
        "loss": ['squared_hinge', 'hinge'], 
        "C": [3.0, 2.0, 1.0, 0.5], 
        "class_weight": [dict(zip(np.unique(y_train), y_train.shape[0] / (2 * np.bincount(y_train)))), None],
        "max_iter": [1000, 1500]
    }

    model_num = 0
    for params in ParameterGrid(param_grid):
        time_start = time()

        print(f"Model num: {model_num}")
        logger.info(
            f"=== New model #{model_num} "
            f"params={params} ==="
        )

        try:
            model = LinearSVC(**params,
                            dual=True,
                            random_state = 42)
            model.fit(X_train, y_train)
        except ValueError as e:
            print(f"Выбранная модель не поддерживает такой набор парметров: {params}")
            continue

        y_pred_test = model.predict(X_test)
        cur_loss_train = hinge_loss(y_train, model.decision_function(X_train))
        cur_loss_test = hinge_loss(y_test, model.decision_function(X_test))
        logger.info(
            f"Model {model_num}| "
            f"f1={f1_score(y_test, y_pred_test):.4f} recall={recall_score(y_test, y_pred_test):.4f} loss_train={cur_loss_train:.4f} "
            f"loss_test={cur_loss_test:.4f}"
        )

        logger.info(f"Model {model_num} | training finished | total_elapsed={(time() - time_start):.2f}s")
        model_num+=1


def cross_validation(X_train, y_train,
                     X_test, y_test, 
                     X_train_subm, X_test_subm, y_train_subm,
                     df_test,
                     n_best_models=10, cv_folds=5, 
                     random_state=42, max_epochs = 1000, 
                     log_versions=[3, 3, 3, 3]):
    # =====================================================
    # ПАРСЕР ЛОГОВ + выбор лучшей модели по CV
    # =====================================================
    LOG_FILES = {
        "SGDClassifier_fit": f"logs/log_reg_train_fit_{log_versions[0]}_SGD.log",
        "SGDClassifier_partial_fit": f"logs/log_reg_train_part_fit_{log_versions[1]}_SGD.log",
        "LogisticRegression": f"logs/log_reg_train_fit_loggin_{log_versions[2]}_LR.log",
        "LinearSVC": f"logs/log_reg_train_fit_loggin_{log_versions[3]}_LSVC.log",
    }

    def clean_params(s):
        s = re.sub(r"np\.int64\((\d+)\)", r"\1", s)
        s = re.sub(r"np\.float64\(([0-9eE.+-]+)\)", r"\1", s)
        return ast.literal_eval(s)

    def parse_fit_log(path):
        pending, records = {}, {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = re.search(r"=== New model #(\d+) params=(.*) ===\s*$", line)
                if m:
                    pending[int(m.group(1))] = clean_params(m.group(2))
                    continue
                m = re.search(r"Model (\d+)\| f1=([\d.]+) recall=([\d.]+)", line)
                if m:
                    num = int(m.group(1))
                    records[num] = {
                        "model_num": num,
                        "params": pending[num],
                        "f1": float(m.group(2)),
                        "recall": float(m.group(3)),
                    }
        return list(records.values())

    def parse_partfit_log(path):
        pending, records = {}, {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = re.search(
                    r"=== New model #(\d+) \| batch_size=(\d+), n_iter_no_change=(\d+), tol=([\d.eE+-]+), params=(.*) ===\s*$",
                    line,
                )
                if m:
                    num = int(m.group(1))
                    pending[num] = {
                        "batch_size": int(m.group(2)),
                        "n_iter_no_change": int(m.group(3)),
                        "tol": float(m.group(4)),
                        "params": clean_params(m.group(5)),
                    }
                    continue
                m = re.search(r"Model (\d+) \| epoch=\d+ idx=\d+ \| f1=([\d.]+) recall=([\d.]+)", line)
                if m:
                    num = int(m.group(1))
                    if num in pending:
                        pending[num]["last_f1"] = float(m.group(2))
                        pending[num]["last_recall"] = float(m.group(3))
                    continue
                m = re.search(r"Model (\d+) \| training finished", line)
                if m:
                    num = int(m.group(1))
                    if num in pending and "last_f1" in pending[num]:
                        rec = dict(pending[num])
                        rec["model_num"] = num
                        rec["f1"] = rec.pop("last_f1")
                        rec["recall"] = rec.pop("last_recall")
                        records[num] = rec
        return list(records.values())

    def top_n(records, n=n_best_models):
        return sorted(records, key=lambda r: (-r["f1"], r["model_num"]))[:n]

    def train_partial_fit_model(params, bs, nic, tol, X, y):
        model = SGDClassifier(**params, random_state=random_state, warm_start=True)
        iters_no_change = 0
        loss_hist = []
        n_samples = X.shape[0]
        for epoch in range(max_epochs):
            for idx in range(0, n_samples, bs):
                start, stop = idx, min(idx + bs, n_samples)
                X_batch, y_batch = X[start:stop], y[start:stop]
                if idx == 0 and epoch == 0:
                    model.partial_fit(X_batch, y_batch, np.unique(y))
                else:
                    model.partial_fit(X_batch, y_batch)
                if idx % (bs * 1000) == 0:
                    if params["loss"] == "log_loss":
                        cur_loss = log_loss(y, model.predict_proba(X))
                    else:
                        cur_loss = hinge_loss(y, model.decision_function(X))
                    loss_hist.append(cur_loss)
                    if iters_no_change >= nic:
                        return model
                    if len(loss_hist) >= 2 and tol > (loss_hist[-2] - loss_hist[-1]):
                        iters_no_change += 1
                    else:
                        iters_no_change = 0
        return model

    def cv_fit(make_model, recs):
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        for rec in recs:
            fold_f1, fold_recall = [], []
            for tr_idx, va_idx in skf.split(X_train, y_train):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = make_model(rec["params"])
                    model.fit(X_train[tr_idx], y_train.iloc[tr_idx])
                pred = model.predict(X_train[va_idx])
                fold_f1.append(f1_score(y_train.iloc[va_idx], pred))
                fold_recall.append(recall_score(y_train.iloc[va_idx], pred))
            rec["mean_f1"] = np.mean(fold_f1)
            rec["mean_recall"] = np.mean(fold_recall)
        return recs

    def cv_partial_fit(recs):
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        for rec in recs:
            fold_f1, fold_recall = [], []
            for tr_idx, va_idx in skf.split(X_train, y_train):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = train_partial_fit_model(
                        rec["params"], rec["batch_size"], rec["n_iter_no_change"], rec["tol"],
                        X_train[tr_idx], y_train.iloc[tr_idx],
                    )
                pred = model.predict(X_train[va_idx])
                fold_f1.append(f1_score(y_train.iloc[va_idx], pred))
                fold_recall.append(recall_score(y_train.iloc[va_idx], pred))
            rec["mean_f1"] = np.mean(fold_f1)
            rec["mean_recall"] = np.mean(fold_recall)
        return recs

    def best_of(recs):
        return max(recs, key=lambda r: (r["mean_f1"], -r["model_num"]))


    def make_report(model_name, best, build_eval_model, build_final_model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eval_model = build_eval_model(best["params"])
            final_model = build_final_model(best["params"])
        pred = eval_model.predict(X_test)
        pred_subm = final_model.predict(X_test_subm)

        submission = pd.DataFrame({"ID": df_test["ID"].values, "label": pred_subm})
        submission.to_csv(f"subs/submission_{model_name}.csv", index=False)

        print("#" * 70)
        print(f"MODEL TYPE: {model_name}")
        print(f"best CV mean_f1 = {best['mean_f1']:.4f}, mean_recall = {best['mean_recall']:.4f}")
        print(f"best params: {best['params']}")
        print("-" * 70)
        print(classification_report(y_test, pred, digits=4))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, pred))
        print(f"submission saved: submission_{model_name}.csv")
        print("#" * 70)
        print()


    # ---------- 1) SGDClassifier (fit) ----------
    try:
        filename = LOG_FILES["SGDClassifier_fit"]
        recs = top_n(parse_fit_log(filename))
        recs = cv_fit(lambda p: SGDClassifier(**p, random_state=random_state, early_stopping=True), recs)
        best = best_of(recs)
        make_report(
            "SGDClassifier_fit",
            best,
            lambda p: SGDClassifier(**p, random_state=random_state, early_stopping=True).fit(X_train, y_train),
            lambda p: SGDClassifier(**p, random_state=random_state, early_stopping=True).fit(X_train_subm, y_train_subm),
        )
    except KeyError:
        print("Нет логов для модели SGDClassifier_fit")

    # ---------- 2) SGDClassifier (partial_fit) ----------
    try:
        recs = top_n(parse_partfit_log(LOG_FILES["SGDClassifier_partial_fit"]))
        recs = cv_partial_fit(recs)
        best = best_of(recs)
        make_report(
            "SGDClassifier_partial_fit",
            best,
            lambda p: train_partial_fit_model(p, best["batch_size"], best["n_iter_no_change"], best["tol"], X_train, y_train),
            lambda p: train_partial_fit_model(p, best["batch_size"], best["n_iter_no_change"], best["tol"], X_train_subm, y_train_subm),
        )
    except KeyError:
        print("Нет логов для модели SGDClassifier_partial_fit")

    # ---------- 3) LogisticRegression ----------
    try:
        recs = top_n(parse_fit_log(LOG_FILES["LogisticRegression"]))
        recs = cv_fit(lambda p: LogisticRegression(**p, random_state=random_state), recs)
        best = best_of(recs)
        make_report(
            "LogisticRegression",
            best,
            lambda p: LogisticRegression(**p, random_state=random_state).fit(X_train, y_train),
            lambda p: LogisticRegression(**p, random_state=random_state).fit(X_train_subm, y_train_subm),
        )
    except KeyError:
        print("Нет логов для модели LogisticRegression")

    # ---------- 4) LinearSVC ----------
    try:
        recs = top_n(parse_fit_log(LOG_FILES["LinearSVC"]))
        recs = cv_fit(lambda p: LinearSVC(**p, dual=True, random_state=random_state), recs)
        best = best_of(recs)
        make_report(
            "LinearSVC",
            best,
            lambda p: LinearSVC(**p, dual=True, random_state=random_state).fit(X_train, y_train),
            lambda p: LinearSVC(**p, dual=True, random_state=random_state).fit(X_train_subm, y_train_subm),
        )
    except KeyError:
        print("Нет логов для модели LinearSVC")
