import numpy as np

class AdalineGD(object):
    """ADAptive LInear NEuron (ADALINE) クラス
    
    勾配降下法を用いて重みを最適化する適応型線形ニューロン

    Args:
        eta (float): 学習率 (0.0 ~ 1.0) (デフォルト: 0.01)
        n_iter (int): 訓練の繰り返し回数 (デフォルト: 50)
        random_state (int): 重みの初期化用の乱数シード (デフォルト: 1)
    
    Attributes:
        w_ (ndarray): 学習後の重みベクトル
                     w_[0]はバイアス項、w_[1:]は各特徴量の重み
        cost_ (list): 各エポックでのコスト関数の値
                     誤差平方和を用いて学習の進捗を監視
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """訓練データに適合させる

        Args:
            X (ndarray): 訓練データ, shape = [n_samples, n_features]
            y (ndarray): 目的変数, shape = [n_samples]

        Returns:
            self: インスタンス自身
        """
        # シード値を元に乱数生成器を初期化
        rgen = np.random.RandomState(self.random_state)
        
        # 重みベクトルを小さな乱数値で初期化
        # 正規分布に従う乱数を生成 (平均0、標準偏差0.01)
        # サイズは特徴量の数 + 1 (バイアス項用)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        
        # コスト関数の履歴を保存するリスト
        self.cost_ = []
        
        # 訓練回数分繰り返す
        for i in range(self.n_iter):
            # 総入力を計算
            net_input = self.net_input(X)
            
            # 活性化関数を適用（ADALINEでは線形なので入力をそのまま出力）
            output = self.activation(net_input)
            
            # 誤差を計算（目的値 - 予測値）
            errors = (y - output)
            
            # 重みの更新（勾配降下法）
            # X.T.dot(errors): 特徴量とエラーの行列積で勾配を計算
            self.w_[1:] += self.eta * X.T.dot(errors)  # 特徴量の重みを更新
            self.w_[0] += self.eta * errors.sum()      # バイアス項を更新
            
            # 誤差平方和をコストとして記録
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
            
        return self
    
    def net_input(self, X):
        """総入力を計算

        Args:
            X (ndarray): 入力データ

        Returns:
            ndarray: 重み付き総和 (w^T x + w_0)
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """活性化関数
        
        ADALINEでは線形活性化関数を使用（入力をそのまま出力）

        Args:
            X (ndarray): 入力データ

        Returns:
            ndarray: 入力データそのまま
        """
        return X
    
    def predict(self, X):
        """クラスラベルを予測

        Args:
            X (ndarray): 入力データ

        Returns:
            ndarray: 予測クラスラベル（1 or -1）
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)