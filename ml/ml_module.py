import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Union

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Opcional: para salvar/carregar modelos, se o pessoal do backend quiser
# import joblib


# ================================
# 1. Funções auxiliares
# ================================

def _validate_dataframe(df: pd.DataFrame, operation: str = "operação") -> None:
    """
    Valida se o DataFrame está em um formato válido.
    
    Raises:
        ValueError: Se o DataFrame for inválido.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Esperado pandas.DataFrame, recebido {type(df).__name__}")
    
    if df.empty:
        raise ValueError(f"DataFrame está vazio. Não é possível realizar {operation}.")
    
    if len(df.columns) == 0:
        raise ValueError(f"DataFrame não possui colunas. Não é possível realizar {operation}.")


def _validate_target_column(df: pd.DataFrame, target_col: str) -> None:
    """
    Valida se a coluna alvo existe no DataFrame.
    
    Raises:
        ValueError: Se a coluna não existir.
    """
    if target_col not in df.columns:
        available_cols = ", ".join(df.columns.tolist()[:10])
        if len(df.columns) > 10:
            available_cols += f", ... (total: {len(df.columns)} colunas)"
        raise ValueError(
            f"Coluna '{target_col}' não encontrada no DataFrame. "
            f"Colunas disponíveis: {available_cols}"
        )


def _validate_test_size(test_size: float) -> None:
    """
    Valida se test_size está no intervalo válido.
    
    Raises:
        ValueError: Se test_size for inválido.
    """
    if not isinstance(test_size, (int, float)):
        raise TypeError(f"test_size deve ser numérico, recebido {type(test_size).__name__}")
    
    if test_size <= 0 or test_size >= 1:
        raise ValueError(f"test_size deve estar entre 0 e 1, recebido {test_size}")


def _validate_model_params(params: Optional[Dict[str, Any]], model_class) -> Dict[str, Any]:
    """
    Valida e filtra parâmetros do modelo, removendo parâmetros inválidos.
    
    Returns:
        Dict com apenas os parâmetros válidos.
    """
    if params is None:
        return {}
    
    if not isinstance(params, dict):
        raise TypeError(f"params deve ser um dicionário, recebido {type(params).__name__}")
    
    valid_params = model_class().get_params().keys()
    filtered_params = {k: v for k, v in params.items() if k in valid_params}
    
    # Avisa sobre parâmetros inválidos (mas não quebra o código)
    invalid_params = set(params.keys()) - set(valid_params)
    if invalid_params:
        import warnings
        warnings.warn(
            f"Parâmetros inválidos ignorados: {', '.join(invalid_params)}. "
            f"Parâmetros válidos: {', '.join(sorted(valid_params))}",
            UserWarning
        )
    
    return filtered_params


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Cria o pré-processador com:
    - StandardScaler para variáveis numéricas
    - OneHotEncoder para variáveis categóricas
    
    Raises:
        ValueError: Se não houver features numéricas ou categóricas.
    """
    _validate_dataframe(X, "construção do pré-processador")
    
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    
    if not numeric_features and not categorical_features:
        raise ValueError(
            "Nenhuma feature numérica ou categórica encontrada. "
            "Verifique os tipos de dados do DataFrame."
        )
    
    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return preprocessor, numeric_features, categorical_features


def regression_metrics(y_true, y_pred):
    """
    Calcula métricas de regressão:
    - R²
    - MAE
    - RMSE
    """
    import numpy as np
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # Calcula RMSE manualmente
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


def classification_metrics(y_true, y_pred, labels=None):
    """
    Calcula métricas de classificação:
    - Accuracy
    - Precision
    - Recall
    - F1
    - Matriz de confusão
    
    Detecta automaticamente se é classificação binária ou multiclasse.
    """
    acc = accuracy_score(y_true, y_pred)
    
    # Detecta número de classes únicas
    n_classes = len(set(y_true) | set(y_pred))
    
    # Para classificação binária, usa average="binary"
    # Para multiclasse, usa average="weighted" (melhor para classes desbalanceadas)
    if n_classes == 2:
        average = "binary"
    else:
        average = "weighted"
    
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm.tolist(),  # para poder virar JSON no backend
        "n_classes": n_classes,
        "average_used": average
    }


# ================================
# 2. Regressão – previsão do valor da compra
# ================================

def get_regressor(model_type: str, params: Optional[Dict[str, Any]] = None):
    """
    Retorna o regressor de acordo com o tipo escolhido.
    
    Args:
        model_type: 'linreg' ou 'rf'
        params: dicionário opcional com hiperparâmetros do modelo
    
    Returns:
        Modelo de regressão configurado
    
    Raises:
        ValueError: Se model_type não for suportado
    """
    if not isinstance(model_type, str):
        raise TypeError(f"model_type deve ser uma string, recebido {type(model_type).__name__}")

    if model_type == "linreg":
        validated_params = _validate_model_params(params, LinearRegression)
        model = LinearRegression(**validated_params)
    elif model_type == "rf":
        validated_params = _validate_model_params(params, RandomForestRegressor)
        model = RandomForestRegressor(**validated_params)
    else:
        raise ValueError(
            f"Tipo de regressor não suportado: '{model_type}'. "
            f"Tipos suportados: 'linreg', 'rf'"
        )

    return model


def train_regression_model(
    df: pd.DataFrame,
    target_col: str,
    model_type: str = "rf",
    params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Treina um modelo de regressão para prever o valor da compra (ou outro alvo contínuo).

    Args:
        df: DataFrame com os dados de treinamento
        target_col: nome da coluna alvo (ex: 'total_amount')
        model_type: 'linreg' ou 'rf'
        params: dicionário opcional com hiperparâmetros (ex: {'n_estimators': 200, 'max_depth': 10})
        test_size: proporção do dataset para teste (entre 0 e 1)
        random_state: seed para reprodutibilidade
    
    Returns:
        Dicionário com modelo treinado, métricas e informações sobre features
    
    Raises:
        ValueError: Se os dados forem inválidos ou insuficientes
    """
    # Validações de entrada
    _validate_dataframe(df, "treinamento de modelo de regressão")
    _validate_target_column(df, target_col)
    _validate_test_size(test_size)
    
    # Verifica se há colunas suficientes após remover o target
    if len(df.columns) < 2:
        raise ValueError(
            f"DataFrame deve ter pelo menos 2 colunas (1 target + 1 feature). "
            f"Encontradas {len(df.columns)} colunas."
        )

    try:
        # Separa X e y
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Verifica se há dados suficientes para treino
        if len(df) < 10:
            raise ValueError(
                f"Dados insuficientes para treinamento. "
                f"Mínimo recomendado: 10 amostras. Encontradas: {len(df)}"
            )
        
        # Verifica se há valores nulos no target
        if y.isna().any():
            n_nulls = y.isna().sum()
            raise ValueError(
                f"Coluna target '{target_col}' contém {n_nulls} valores nulos. "
                f"Remova ou preencha esses valores antes do treinamento."
            )

        # Pré-processador
        preprocessor, num_cols, cat_cols = build_preprocessor(X)

        # Modelo
        regressor = get_regressor(model_type, params)

        # Pipeline completo
        model = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", regressor)
        ])

        # Split treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Treino
        model.fit(X_train, y_train)

        # Avaliação
        y_pred = model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)

        # Retorna algumas predições de exemplo para visualização (primeiras 20)
        sample_size = min(20, len(y_test))
        sample_indices = list(range(sample_size))
        y_test_sample = y_test.iloc[sample_indices].tolist() if hasattr(y_test, 'iloc') else y_test[sample_indices].tolist()
        y_pred_sample = y_pred[sample_indices].tolist()

        result = {
            "model_type": model_type,
            "model": model,
            "target_col": target_col,
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
            "metrics": metrics,
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test),
            "y_test_sample": y_test_sample,
            "y_pred_sample": y_pred_sample
        }

        return result
    
    except Exception as e:
        raise RuntimeError(
            f"Erro ao treinar modelo de regressão: {str(e)}"
        ) from e


def predict_regression(model_pipeline: Pipeline, new_data: pd.DataFrame):
    """
    Faz previsões de regressão em novos dados.
    
    Args:
        model_pipeline: Pipeline treinado (deve conter 'preprocess' e 'model')
        new_data: DataFrame com os dados para predição (deve ter as mesmas colunas de X usadas no treino)
    
    Returns:
        Array numpy com as previsões
    
    Raises:
        ValueError: Se new_data estiver vazio ou não tiver as colunas esperadas
    """
    _validate_dataframe(new_data, "predição de regressão")
    
    if not isinstance(model_pipeline, Pipeline):
        raise TypeError(
            f"model_pipeline deve ser um Pipeline do sklearn, "
            f"recebido {type(model_pipeline).__name__}"
        )
    
    # Verifica se o pipeline tem os steps esperados
    if "preprocess" not in model_pipeline.named_steps:
        raise ValueError("Pipeline deve conter o step 'preprocess'")
    if "model" not in model_pipeline.named_steps:
        raise ValueError("Pipeline deve conter o step 'model'")
    
    try:
        preds = model_pipeline.predict(new_data)
        return preds
    except Exception as e:
        raise RuntimeError(
            f"Erro ao fazer predições de regressão: {str(e)}. "
            f"Verifique se new_data tem as mesmas colunas usadas no treinamento."
        ) from e


# ================================
# 3. Classificação – prever cliente recorrente vs novo
# ================================

def get_classifier(model_type: str, params: Optional[Dict[str, Any]] = None):
    """
    Retorna o classificador de acordo com o tipo escolhido.
    
    Args:
        model_type: 'logreg', 'rf' ou 'knn'
        params: dicionário opcional com hiperparâmetros do modelo
    
    Returns:
        Modelo de classificação configurado
    
    Raises:
        ValueError: Se model_type não for suportado
    """
    if not isinstance(model_type, str):
        raise TypeError(f"model_type deve ser uma string, recebido {type(model_type).__name__}")

    if model_type == "logreg":
        validated_params = _validate_model_params(params, LogisticRegression)
        # Garante max_iter=1000 por padrão para evitar warnings
        if "max_iter" not in validated_params:
            validated_params["max_iter"] = 1000
        model = LogisticRegression(**validated_params)

    elif model_type == "rf":
        validated_params = _validate_model_params(params, RandomForestClassifier)
        model = RandomForestClassifier(**validated_params)

    elif model_type == "knn":
        validated_params = _validate_model_params(params, KNeighborsClassifier)
        model = KNeighborsClassifier(**validated_params)

    else:
        raise ValueError(
            f"Tipo de classificador não suportado: '{model_type}'. "
            f"Tipos suportados: 'logreg', 'rf', 'knn'"
        )

    return model


def train_classification_model(
    df: pd.DataFrame,
    target_col: str,
    positive_label: Optional[Union[str, int]] = None,
    model_type: str = "rf",
    params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Treina um modelo de classificação binária (ex: cliente recorrente vs novo).

    Args:
        df: DataFrame com os dados de treinamento
        target_col: nome da coluna alvo (ex: 'customer_type')
        positive_label: opcional, rótulo que você considera como "1" (ex: 'Returning').
                        Se None, o LabelEncoder vai tratar automaticamente.
        model_type: 'logreg', 'rf' ou 'knn'
        params: dicionário opcional com hiperparâmetros
        test_size: proporção do dataset para teste (entre 0 e 1)
        random_state: seed para reprodutibilidade
    
    Returns:
        Dicionário com modelo treinado, métricas e informações sobre features
    
    Raises:
        ValueError: Se os dados forem inválidos ou insuficientes
    """
    # Validações de entrada
    _validate_dataframe(df, "treinamento de modelo de classificação")
    _validate_target_column(df, target_col)
    _validate_test_size(test_size)
    
    # Verifica se há colunas suficientes após remover o target
    if len(df.columns) < 2:
        raise ValueError(
            f"DataFrame deve ter pelo menos 2 colunas (1 target + 1 feature). "
            f"Encontradas {len(df.columns)} colunas."
        )

    try:
        # Separa X e y
        X = df.drop(columns=[target_col])
        y = df[target_col].copy()

        # Verifica se há dados suficientes para treino
        if len(df) < 10:
            raise ValueError(
                f"Dados insuficientes para treinamento. "
                f"Mínimo recomendado: 10 amostras. Encontradas: {len(df)}"
            )
        
        # Verifica se há valores nulos no target
        if y.isna().any():
            n_nulls = y.isna().sum()
            raise ValueError(
                f"Coluna target '{target_col}' contém {n_nulls} valores nulos. "
                f"Remova ou preencha esses valores antes do treinamento."
            )
        
        # Verifica número de classes únicas
        unique_classes = y.unique()
        n_classes = len(unique_classes)
        
        if n_classes < 2:
            raise ValueError(
                f"Classificação requer pelo menos 2 classes. "
                f"Encontrada(s) {n_classes} classe(s) única(s): {unique_classes.tolist()}"
            )

        # LabelEncoder para transformar categorias em 0/1
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Garante que positive_label seja mapeado para classe 1
        if positive_label is not None:
            # Verifica se o positive_label existe em y
            if positive_label not in unique_classes:
                raise ValueError(
                    f"positive_label '{positive_label}' não encontrado na coluna '{target_col}'. "
                    f"Classes disponíveis: {unique_classes.tolist()}"
                )
            
            # Se o positive_label não for mapeado para 1, inverte o encoding
            positive_encoded = label_encoder.transform([positive_label])[0]
            if positive_encoded != 1:
                # Inverte: 0 vira 1, 1 vira 0
                y_encoded = 1 - y_encoded
                # Atualiza o label_encoder para refletir a inversão
                # Criamos um novo encoder com classes invertidas
                classes_original = label_encoder.classes_
                label_encoder.classes_ = classes_original[::-1]

        # Pré-processador
        preprocessor, num_cols, cat_cols = build_preprocessor(X)

        # Classificador
        classifier = get_classifier(model_type, params)

        # Pipeline completo
        model = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", classifier)
        ])

        # Split treino/teste com stratify (garante proporção de classes)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
            )
        except ValueError as e:
            # Se stratify falhar (ex: classe com apenas 1 amostra), tenta sem stratify
            import warnings
            warnings.warn(
                f"Não foi possível usar stratify no split: {str(e)}. "
                f"Usando split sem stratify.",
                UserWarning
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_state
            )

        # Treino
        model.fit(X_train, y_train)

        # Avaliação
        y_pred_encoded = model.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        # Calcula probabilidades se o modelo suportar
        y_proba = None
        if hasattr(model.named_steps['model'], 'predict_proba'):
            try:
                y_proba_encoded = model.named_steps['model'].predict_proba(X_test)
                # Para classificação binária, pega a probabilidade da classe positiva (índice 1)
                if y_proba_encoded.shape[1] == 2:
                    y_proba = y_proba_encoded[:, 1]
                else:
                    # Para multiclasse, pega a probabilidade máxima
                    y_proba = y_proba_encoded.max(axis=1)
            except:
                pass
        
        # Converte y_test de volta para valores originais para métricas
        y_test_original = label_encoder.inverse_transform(y_test)
        metrics = classification_metrics(y_test_original, y_pred, labels=label_encoder.classes_)

        # Retorna algumas predições de exemplo para visualização (primeiras 20)
        sample_size = min(20, len(y_test_original))
        sample_indices = list(range(sample_size))
        y_test_sample = y_test_original[sample_indices].tolist() if hasattr(y_test_original, 'tolist') else list(y_test_original[sample_indices])
        y_pred_sample = y_pred[sample_indices].tolist() if hasattr(y_pred, 'tolist') else list(y_pred[sample_indices])
        y_proba_sample = y_proba[sample_indices].tolist() if y_proba is not None else None

        result = {
            "model_type": model_type,
            "model": model,
            "label_encoder": label_encoder,
            "target_col": target_col,
            "classes_": label_encoder.classes_.tolist(),
            "positive_label": positive_label,
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
            "metrics": metrics,
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test),
            "n_classes": n_classes,
            "y_test_sample": y_test_sample,
            "y_pred_sample": y_pred_sample,
            "y_proba_sample": y_proba_sample
        }

        return result
    
    except Exception as e:
        raise RuntimeError(
            f"Erro ao treinar modelo de classificação: {str(e)}"
        ) from e


def predict_classification(
    model_pipeline: Pipeline,
    new_data: pd.DataFrame,
    label_encoder: Optional[LabelEncoder] = None,
    return_proba: bool = True
):
    """
    Faz previsões de classificação em novos dados.
    
    Args:
        model_pipeline: Pipeline treinado (deve conter 'preprocess' e 'model')
        new_data: DataFrame com os dados para predição (deve ter as mesmas colunas de X usadas no treino)
        label_encoder: Opcional, LabelEncoder usado no treinamento para converter labels
        return_proba: Se True, retorna também as probabilidades (apenas para modelos que suportam)
    
    Returns:
        Dicionário com:
        - y_pred_encoded: previsões codificadas (0/1)
        - y_pred_labels: previsões como labels originais (se label_encoder fornecido)
        - y_proba: probabilidades da classe positiva (se disponível)
    
    Raises:
        ValueError: Se new_data estiver vazio ou não tiver as colunas esperadas
    """
    _validate_dataframe(new_data, "predição de classificação")
    
    if not isinstance(model_pipeline, Pipeline):
        raise TypeError(
            f"model_pipeline deve ser um Pipeline do sklearn, "
            f"recebido {type(model_pipeline).__name__}"
        )
    
    # Verifica se o pipeline tem os steps esperados
    if "preprocess" not in model_pipeline.named_steps:
        raise ValueError("Pipeline deve conter o step 'preprocess'")
    if "model" not in model_pipeline.named_steps:
        raise ValueError("Pipeline deve conter o step 'model'")
    
    try:
        y_pred = model_pipeline.predict(new_data)

        # Tenta obter probabilidades se solicitado e disponível
        y_proba = None
        if return_proba:
            model_step = model_pipeline.named_steps["model"]
            if hasattr(model_step, "predict_proba"):
                try:
                    proba_matrix = model_pipeline.predict_proba(new_data)
                    # Para classificação binária, pega a probabilidade da classe 1
                    if proba_matrix.shape[1] == 2:
                        y_proba = proba_matrix[:, 1]
                    else:
                        # Para multiclasse, retorna a matriz completa
                        y_proba = proba_matrix.tolist()
                except Exception as e:
                    import warnings
                    warnings.warn(
                        f"Não foi possível obter probabilidades: {str(e)}",
                        UserWarning
                    )

        # Converte labels se encoder fornecido
        if label_encoder is not None:
            if not isinstance(label_encoder, LabelEncoder):
                raise TypeError(
                    f"label_encoder deve ser um LabelEncoder, "
                    f"recebido {type(label_encoder).__name__}"
                )
            try:
                labels = label_encoder.inverse_transform(y_pred)
            except Exception as e:
                raise ValueError(
                    f"Erro ao converter labels usando label_encoder: {str(e)}. "
                    f"Verifique se o encoder corresponde ao modelo treinado."
                ) from e
        else:
            labels = y_pred

        return {
            "y_pred_encoded": y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
            "y_pred_labels": labels.tolist() if hasattr(labels, 'tolist') else labels,
            "y_proba": y_proba.tolist() if y_proba is not None and hasattr(y_proba, 'tolist') else y_proba
        }
    
    except Exception as e:
        raise RuntimeError(
            f"Erro ao fazer predições de classificação: {str(e)}. "
            f"Verifique se new_data tem as mesmas colunas usadas no treinamento."
        ) from e


# ================================
# 4. Treinamento Dinâmico / Função geral
# ================================

def train_all_models(
    df: pd.DataFrame,
    target_reg: str,
    target_clf: str,
    reg_model_type: str = "rf",
    clf_model_type: str = "rf",
    reg_params: Optional[Dict[str, Any]] = None,
    clf_params: Optional[Dict[str, Any]] = None,
    positive_label: Optional[Union[str, int]] = None,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Função geral que treina:
    - um modelo de regressão para target_reg
    - um modelo de classificação para target_clf

    Essa função pode ser chamada toda vez que o usuário subir um novo CSV
    → garante o "treinamento dinâmico" pedido no enunciado.
    
    Args:
        df: DataFrame com os dados de treinamento
        target_reg: nome da coluna alvo para regressão
        target_clf: nome da coluna alvo para classificação
        reg_model_type: tipo de modelo de regressão ('linreg' ou 'rf')
        clf_model_type: tipo de modelo de classificação ('logreg', 'rf' ou 'knn')
        reg_params: parâmetros opcionais para o modelo de regressão
        clf_params: parâmetros opcionais para o modelo de classificação
        positive_label: rótulo positivo para classificação (opcional)
        test_size: proporção do dataset para teste (entre 0 e 1)
        random_state: seed para reprodutibilidade
    
    Returns:
        Dicionário com resultados de ambos os modelos treinados
    
    Raises:
        ValueError: Se os dados forem inválidos ou as colunas não existirem
    """
    # Validações básicas
    _validate_dataframe(df, "treinamento de todos os modelos")
    _validate_target_column(df, target_reg)
    _validate_target_column(df, target_clf)
    
    # Verifica se as colunas target são diferentes
    if target_reg == target_clf:
        raise ValueError(
            f"As colunas target para regressão e classificação devem ser diferentes. "
            f"Recebido: '{target_reg}' para ambos."
        )
    
    try:
        reg_result = train_regression_model(
            df=df,
            target_col=target_reg,
            model_type=reg_model_type,
            params=reg_params,
            test_size=test_size,
            random_state=random_state
        )

        clf_result = train_classification_model(
            df=df,
            target_col=target_clf,
            positive_label=positive_label,
            model_type=clf_model_type,
            params=clf_params,
            test_size=test_size,
            random_state=random_state
        )

        return {
            "regression": reg_result,
            "classification": clf_result
        }
    
    except Exception as e:
        raise RuntimeError(
            f"Erro ao treinar todos os modelos: {str(e)}"
        ) from e

    # Exemplo se quiser salvar os modelos em disco:
    # import joblib
    # joblib.dump(reg_result["model"], "regression_model.pkl")
    # joblib.dump(clf_result["model"], "classification_model.pkl")
    # joblib.dump(clf_result["label_encoder"], "classification_label_encoder.pkl")
