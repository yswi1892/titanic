import pickle

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from flaml import AutoML
from sklearn import tree, metrics
from dtreeviz.trees import dtreeviz


def make_df():

    # データの取得
    train = pd.read_csv(Path('data', 'ti.train.csv'), index_col='ID')
    test = pd.read_csv(Path('data', 'ti.test.csv'), index_col='ID')

    # 学習検証列を追加
    train['TrainTest'] = 'train'
    test['TrainTest'] = 'test'

    # 学習検証を結合
    df = pd.concat([train, test])

    return df


def feature_engineering_dt():

    # DFを作成
    df = make_df()

    # 性別を数値に置換
    df['sex'].replace({'male': 0, 'female': 1}, inplace=True)

    # 年齢の欠損値を中央値で埋め
    df['age'].fillna(df['age'].median(), inplace=True)

    # 出発港を最頻値で埋め
    mode = df['embarked'].mode()[0]
    df['embarked'].fillna(mode, inplace=True)

    # 出発港をOne-Hot Encoding
    embarked_enc = pd.get_dummies(df['embarked'], prefix='embarked')
    df = pd.concat([df, embarked_enc], axis=1)
    df.drop('embarked', axis=1, inplace=True)

    return df


def feature_engineering_flaml():

    # DFを作成
    df = make_df()

    # embarkedを最頻値で埋める
    mode = df['embarked'].mode()[0]
    df['embarked'].fillna(mode, inplace=True)

    # ageを中央値で埋める
    df['age'].fillna(df['age'].median(), inplace=True)

    # クラスと出発港でグループ化し、料金の中央値を算出
    gb = ['pclass', 'embarked']
    fare_gb = df.groupby(gb)['fare'].median().rename('fare_gb')
    df = df.merge(fare_gb, on=gb, how='left').set_index(df.index)

    # クラスと出発港でグループ化し、料金/料金頻度を算出
    fare_counts = df['fare'].value_counts().rename('fare_counts')
    df = df.merge(fare_counts, left_on='fare', right_index=True)
    df['fare_counts_ratio'] = df['fare'] / df['fare_counts']

    # 家族サイズを追加
    df['family_size'] = df['sibsp'] + df['parch'] + 1

    return df


def train_test_split(df):

    # X、Yに分離
    x, y = df.drop('survived', axis=1), df['survived']

    # 学習、検証に分離
    train_x = x.loc[x['TrainTest'] == 'train'].drop('TrainTest', axis=1)
    train_y = y.loc[x['TrainTest'] == 'train']
    test_x = x.loc[x['TrainTest'] == 'test'].drop('TrainTest', axis=1)

    return train_x, train_y, test_x


def decision_tree():

    # 特徴量エンジニアリングを実施
    df = feature_engineering_dt()

    # 学習、検証に分離
    train_x, train_y, _ = train_test_split(df)

    # 決定木インスタンスを作成
    model = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=5)

    # 決定木を実行
    model.fit(train_x, train_y)

    # dtreevizで描画
    viz = dtreeviz(
        model, train_x, train_y, target_name='survived',
        feature_names=train_x.columns,
        class_names=[False, True])

    # 決定木を保存
    viz.save('titanic.svg')


def run_flaml():

    # 特徴量エンジニアリングを実施
    df = feature_engineering_flaml()

    # 学習、検証に分離
    train_x, train_y, test_x = train_test_split(df)

    # FLAMLのパラメータ設定
    automl_settings = {
        'time_budget': 5,
        'metric': 'roc_auc',
        'task': 'classification',
        'estimator_list': ['lgbm'],
        'n_splits': 10,
        'seed': 3,
    }

    # FLAMLで学習
    model = AutoML()
    model.fit(X_train=train_x, y_train=train_y, **automl_settings)

    # モデルを保存
    with open(Path('model', 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    # 最適予測器、パラメータを表示
    print(model.best_estimator)
    print(model.best_config)

    # 検証データを予測
    test_y_pred = pd.Series(
        model.predict_proba(test_x)[:, 1], index=test_x.index, name='y')

    # 生存条件
    survived = (test_x['sex'] == 'female') & (test_x['pclass'] != 3)
    survived1 = survived & (test_x['fare'] >= 28.86)
    survived2 = survived & (test_x['fare'] <= 56) & (test_x['parch'] >= 1)

    # 死亡条件
    died = \
        (test_x['sex'] == 'male') & (test_x['age'] >= 14) & \
        (test_x['pclass'] != 1) & (test_x['embarked'] != 'C') & \
        (test_x['parch'] >= 1)

    # 予測結果を上書き
    test_y_pred.loc[survived1 + survived2] = 1
    test_y_pred.loc[died] = 0

    # 予測結果を出力
    test_y_pred.to_csv(Path('data', 'submit.csv'))

    # 結果を辞書化
    results = {
        'train_x': train_x,
        'train_y': train_y,
        'test_x': test_x,
        'test_y_pred': test_y_pred,
        'model': model,
    }

    return results


def make_importance(results):

    # 最適モデルを取得
    best_model = results['model'].model.estimator

    # 学習のXを取得
    train_x = results['train_x']

    # SHAP値を取得
    shap_values = pd.DataFrame(
        best_model.predict(train_x, pred_contrib=True)[:, :-1],
        index=train_x.index, columns=train_x.columns
    )
    # 重要度を算出
    importance = shap_values.abs().mean().sort_values()

    # 重要度をグラフ化
    plt.style.use('ggplot')
    plt.barh(importance.index, importance)
    plt.xlabel('Importance: mean(|SHAP values|)')

    # グラフを保存
    plt.savefig(Path('img', 'importance.png'))

    # 重要度を描画
    plt.show()


def make_roc(results):

    # 予測結果を取得
    test_y_pred = results['test_y_pred']

    # 正答を取得
    path = Path('data', 'ti.answer.csv')
    answer = pd.read_csv(path, index_col='PassengerId')
    test_y = answer['Survived'].loc[test_y_pred.index]

    # FPR、TPR、AUCを取得
    fpr, tpr, thresholds = metrics.roc_curve(test_y, test_y_pred)
    auc = metrics.auc(fpr, tpr)

    # RCO曲線を作成
    plt.plot(fpr, tpr, label='ROC curve (area = %.3f)' % auc)

    # グラフを整形
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.legend()
    plt.grid()

    # グラフを保存
    plt.savefig(Path('img', 'roc.png'))

    # グラフを描画
    plt.show()


def main():

    # # 決定木を作成
    # decision_tree()

    # FLAMLを実行
    results = run_flaml()

    # 重要度を描画
    make_importance(results)

    # ROC曲線を取得
    make_roc(results)


if __name__ == '__main__':
    main()
