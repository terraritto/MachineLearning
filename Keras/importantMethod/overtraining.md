# Important Point for Overtraining
* 訓練データを増やす
* ネットワークのキャパシティを減らす  
つまりは適切なDense(層)を決めようね！！ってこと
* 重みを正規化する  
regularizerにl1,l2,l1l2などを設定してやる
* ドロップアウトを追加する  
原理としてはノイズがあっても大丈夫にするようなもん  
layers.Dropoutで使える、値は0.2~0.5くらいがよい。