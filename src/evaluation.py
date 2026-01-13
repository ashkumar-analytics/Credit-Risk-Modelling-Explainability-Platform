from sklearn.metrics import roc_auc_score, classification_report

def evaluate_model(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    return {
        "AUC": auc,
        "Report": classification_report(y_test, model.predict(X_test))
    }
