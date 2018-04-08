#!/usr/bin/env perl
use PDL;
use PDL::Ufunc;
use PDL::NiceSlice;

$TRAIN_DATA_SIZE   = 50;    # 150個のデータのうちTRAIN_DATA_SIZE個を訓練データとして使用。残りは教師データとして使用。
$HIDDEN_LAYER_SIZE = 6;     # 中間層(隠れ層)のサイズ(今回は中間層は1層なのでスカラー)
$LEARNING_RATE     = 0.1;   # 学習率
$ITERS_NUM         = 1000;  # 繰り返し回数
$DELTA             = 0.01;

# データを読み込み
# デフォルトで'#'の行をを飛ばすようにはなってないので、一旦外の処理でコメント行を削除してから読み込み
$x     = rcols("iris.csv",[0..3], { DEFTYPE=>float, COLSEP=>"," })->transpose;
$raw_t = rcols("iris.csv",[4], { DEFTYPE=>float, COLSEP=>"," });
$onehot_t = float zeroes(3,150);

for (0 .. shape($onehot_t)->at(1)-1){
    $onehot_t($raw_t->at($_,0) , $_) .= float 1;
}

$train_x = $x(:, :$TRAIN_DATA_SIZE-1);
$train_t = $onehot_t(:, :$TRAIN_DATA_SIZE-1);
$test_x  = $x(:, $TRAIN_DATA_SIZE:);
$test_t  = $onehot_t(:, $TRAIN_DATA_SIZE:);

# 重みとバイアスの初期化
$W1 = grandom($HIDDEN_LAYER_SIZE, 4) x sqrt(2/4);
$W2 = grandom(3, $HIDDEN_LAYER_SIZE) x sqrt(2/$HIDDEN_LAYER_SIZE);
$b1 = zeroes($HIDDEN_LAYER_SIZE);
$b2 = zeroes(3);

# テストデータの結果
$test_y = forward($test_x);
print float($test_y->maximum_ind == $test_t->maximum_ind)->sum, '/', 150- $TRAIN_DATA_SIZE, "\n";

for (1 .. $ITERS_NUM){

    # 順伝搬withデータ保存
    $y1 = $train_x x $W1 + $b1;
    $y2 = relu($y1);
    $train_y = softmax($y2 x $W2 + $b2);

    # 損失関数計算
    $L = cross_entropy_error($train_y, $train_t);

    print $L,"\n" if $_ % 100 == 0;

    # 勾配計算
    # 計算グラフで求めた式を使用
    $a1 = ($train_y - $train_t) / $TRAIN_DATA_SIZE;
    $b2_gradient = dsumover $a1->transpose;
    $W2_gradient = $y2->transpose x $a1;
    $a2 = $a1 x $W2->transpose;
    $a2 = $a2 * (pdl [$y1 > 0]);

    $b1_gradient = (dsumover $a2->transpose)->clump(-1);
    $W1_gradient = $train_x->transpose x $a2->clump(1..2);

    # パラメータ更新
    $W1 = $W1  - $LEARNING_RATE * $W1_gradient;
    $W2 = $W2  - $LEARNING_RATE * $W2_gradient;
    $b1 = $b1  - $LEARNING_RATE * $b1_gradient;
    $b2 = $b2  - $LEARNING_RATE * $b2_gradient;

}

# 最終訓練データのL値
$L = cross_entropy_error(forward($train_x), $train_t);
print $L,"\n";

# テストデータの結果
$test_y = forward($test_x);
print pdl($test_y->maximum_ind == $test_t->maximum_ind)->sum, '/', 150- $TRAIN_DATA_SIZE, "\n";


# 以下、計算に必要な各種関数
sub relu{
    my $x = shift @_; 
    my $idx = $x >= 0.0;
    return $x * $idx;
}

sub softmax{
    my $x = shift @_;
    # 後ろの除算のoverflow対策で最大値を引いておく
    $x = exp $x - max($x);

    if($x->getndims == 1){
        $sum = dsumover $x;
        return $x / $sum;
    }elsif($x->getndims == 2){
        $sum = dsumover $x;
        return $x / $sum->transpose;
    }else{
        die "Dimensions are not identical!";
    }
}

sub cross_entropy_error{
    my ($y, $t) = @_;
    my $z = (shape $y)->at(1);

    if ( $y->getndims != $t->getndims ){
        die "Dimensions are not identical!";
    }elsif( $y->getndims == 1 ){
        return -($t * log($y))->sum;
    }elsif( $y->getndims == 2 ){
        return -($t * log($y))->sum / $z;
    }else{
        die "Dimension exceeds 2";
    }

}

sub forward{
    my $x = shift @_;

    my $y1 = relu($x x $W1 + $b1);
    my $y2 = softmax($y1 x $W2 + $b2);

    return $y2;
}