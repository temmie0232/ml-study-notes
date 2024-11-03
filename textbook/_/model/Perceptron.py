import numpy as np

class Perceptron(object):
    """パーセプトロンクラス
    
    二値分類のための単純な線形分類器。オンライン学習アルゴリズムを使用。

    Args:
        eta (float): 学習率 (0.0 ~ 1.0) (デフォルト: 0.01)
                    各重み更新の大きさを制御するパラメータ
        n_iter (int): 訓練の繰り返し回数 (デフォルト: 50)
                     データセット全体の学習を繰り返す回数（エポック数）
        random_state (int): 重みの初期化用の乱数シード (デフォルト: 1)
                          再現性を確保するために使用
    
    Attributes:
        w_ (ndarray): 学習後の重みベクトル
                     w_[0]はバイアス項、w_[1:]は各特徴量の重み
        errors_ (list): 各エポックでの誤分類の数
                       学習の収束状況を確認するために使用
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """訓練データに適合させる（学習を行う）

        Args:
            X (ndarray): 訓練データ, shape = [n_samples, n_features]
                        n_samples: データ点の数
                        n_features: 特徴量の数
            y (ndarray): 目的変数（正解ラベル）, shape = [n_samples]
                        クラスラベルは1か-1

        Returns:
            self: インスタンス自身
        """
        # 乱数生成器を初期化（再現性のため）
        rgen = np.random.RandomState(self.random_state)
        
        # 重みベクトルを小さな乱数値で初期化
        # 正規分布（平均0、標準偏差0.01）に従う乱数を生成
        # サイズは特徴量の数 + 1（バイアス項用）
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        
        # エポックごとの誤分類数を記録するリスト
        self.errors_ = []
        
        # 訓練回数（エポック数）分繰り返す
        for _ in range(self.n_iter):
            errors = 0
            
            # 各訓練データに対してオンライン学習を行う
            for xi, target in zip(X, y):
                
                # (※1) 重みの更新量を計算
                # target: 正解ラベル（1 or -1）
                # predict(xi): 現在の重みでの予測値
                # 更新量 = 学習率 * (正解ラベル - 予測値)
                update = self.eta * (target - self.predict(xi))
                
                # (※2) 特徴量の重みを更新
                # 更新量とデータ点の積で重みを調整
                self.w_[1:] += update * xi
                
                # バイアス項を更新
                # バイアス項は常に1を掛けるため、更新量をそのまま加算
                self.w_[0] += update
                
                # 誤分類があった場合（更新量が0でない場合）
                # エラーカウントを増やす
                errors += int(update != 0.0)
            
            # 現在のエポックでの誤分類数を記録
            # 誤分類数が0になれば学習完了
            self.errors_.append(errors)
        
        return self
    
    def net_input(self, X):
        """(※3) 入力層での総入力を計算

        Args:
            X (ndarray): 入力データ

        Returns:
            float: 重み付き総和 (w^T x + w_0)
        """
        # 特徴量と重みの内積 + バイアス項
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """(※4) クラスラベルを予測

        Args:
            X (ndarray): 入力データ

        Returns:
            ndarray: 予測クラスラベル（1 or -1）
        """
        # ステップ関数
        # 総入力が0以上なら1、0未満なら-1を返す
        return np.where(self.net_input(X) >= 0.0, 1, -1)

        
"""
=====================================================
※1

場合1: 正しく分類できていない場合（target=1, 予測=-1）
prediction = -1
update = 0.01 * (1 - (-1))     = 0.01 * 2 = 0.02
→ 正の方向に更新（重みを増やす）

場合2: 正しく分類できていない場合（target=-1, 予測=1）
prediction = 1
update = 0.01 * (-1 - 1)       = 0.01 * -2 = -0.02
→ 負の方向に更新（重みを減らす）

場合3: 正しく分類できている場合（target=1, 予測=1）
prediction = 1
update = 0.01 * (1 - 1)        = 0 
→ 更新なし


=====================================================
※2

self.w_ = [0.1, 0.2, 0.3]    [バイアス項, 重み1, 重み2]
xi = [2, 3]                  入力データ [特徴量1, 特徴量2]
update = 0.02                先ほど計算した更新量

self.w_[1:] は [0.2, 0.3]  （バイアス以外の重み）
update * xi は [0.02 * 2, 0.02 * 3] = [0.04, 0.06]

更新後：
self.w_[1:] += [0.04, 0.06]  [0.24, 0.36]


=====================================================
※3

X = [2, 3]           # 入力データ（特徴量が2つ）
self.w_ = [0.1, 0.2, 0.3]  # [バイアス, 重み1, 重み2]

1. np.dot(X, self.w_[1:])
   = X[0]*w[1] + X[1]*w[2]
   = 2*0.2 + 3*0.3
   = 0.4 + 0.9
   = 1.3

2. + self.w_[0]
   = 1.3 + 0.1
   = 1.4


np.dot(X, self.w_[1:])

各入力値と対応する重みの積の和
特徴量の重要度を考慮した計算


+ self.w_[0]

バイアス項を加算
決定境界の位置を調整


=====================================================
※4

np.where(condition)

条件を満たす要素のインデックスを返します


np.where(condition, x, y)

条件が真の場合はx、偽の場合はyを返します
xとyは配列やスカラー値が使えます


多次元配列にも対応しています

返されるインデックスは座標のタプルとなります

"""