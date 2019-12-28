# Important Point for evaluation
* データは毎回シャッフルする
* Arrow Of Time  
得られたデータの時間に着目する(株価や天気)  
test data は training data よりも時間的に未来のものが望ましい。
* Redundancy of data  
同じデータが training data, Validation data の両方で出てくるのは最悪なので、この二つのデータがしっかり互いに素になってるか確認する。