０．コーディングの大前提
⑴ コードは他に人が最短で理解できるように書かなければならない
⑵ スコープが小さければ，変数名とかちゃんとしなくてもよい
⑶ 一貫性のあるコーディングをする
（重要度順に並べるとか，アルファベット順に並べるとか）
⑷ 関数を定義する時，引数をちゃんと書いといたほうがbetter．

（http://www.lifewithpython.com/2013/01/pep-8-style-guide-for-python-code.html）
０．命名規則
デフォルト変数　_df
ベクトル変数　_vec
インデックス　idx
クラス名はクラス名は大文字から
改行後のインデントはそろえる．
クラス定義ブロックの前後は２行空行を入れる
縦のラインを合わせるために空白スペースは入れないこと
コメント文は極力英語で
１行72文字以内に収める
複数行になるときは，インデントを４つ下げる
foo = long_function_name(
var_one, var_two,
var_three, var_four)
イテレータが複数ある場合は，説明的な名前をつけたほうがbetter
変数の単位をつけるのもよい（time64sとか）

１．使える変数名集
exclude・select・value

１．module毎に説明書きを入れる（わかりにくいやつは）
‘''
‘''
↑これ
（http://blog.amedama.jp/entry/2015/10/15/212527）
このときに入力は何で，出力は何かをメモ書き残しておいた方がよい．
２．普通にDropboxに共有させないようにして，Githubで管理する方が良い
３．@staticmethodと@classmethod
@staticmethod：クラス内部で完結するもの（文頭にアンダーバー２つ，つける(__)）
@classmethod：クラス外部で用いるもの
４．for文はキーの場合はkeyで設定する．インデックスの場合はiで設定する．また単にループさせる時は_でやる
５．変数として用いるものは大文字で設定する
６．欠損値に対しては”NA”で埋める
７．カラムに存在する特定のカラム名を削除する
cols = df.columns.tolist()
cols.remove("DISPFROM")
cols.remove("DISPEND")
８．ソートしてインデックスを修正する．
train_coupon_df = train_coupon_df.sort(columns=["DISFROM"]).reset_index(drop=True)
９．データの選択
A[[True,False]]・・これは１行目を選択する
B[“column1”]・・これは１列目を選択する
１０．メッセージを蓄積させないようにしたい場合
　　　　 if c[0] % 100 ==0: とsys.stdout.flush()で出力を更新しながら提出できる．
print "load users: %d/%d\r" % (i, len(user_df)),
sys.stdout.flush()
１１．verboseは表示させるかどうか
１２．配列のインデックスも同時にfor文にかける　　for i,user in enumerate(self.users):
１３．複数の値をfor分にかける　　　for (a, b) in zip(list1, list2)
１４．長さの異なるデータ同士の保存方法
users.append({"user": user_vec[i],
"coupon_ids": row_ids,
"valid_coupon_ids": valid_row_ids})
１５．全てのデータの時系列情報を昇順にして，インデックスもリセットする方が統一的でbetter
１６．日常的に変更するパラメータ値に関しては大文字にする方がbetterかも
１７．open("models/{}_{}.pkl".format(model_name, args.seed)，文字列に変数を加える際の書き方．
１８．何かペアにして扱いたい時．A = zip(pred, rec["coupon_ids”]) A[0]でアクセスできる
２０．logger使った方が良いかなー．
２１．map(メソッド名,変数１,変数２) = apply(lamda x: x )
２２．１対１対応させたい時は，dict形式でデータを格納するといい．（長さが異なる時も）
　　　紐付けする時は，dfTrain["qid"] = map(lambda q: qid_dict[q], dfTrain["query"])
　　　んで，mapを使うべし．（最初に関数（やりたいこと），後にアクセスする配列）
２３．groupbyのcountを指定したカラム（”SEX_ID”）に対して行う　user_df.groupby(["SEX_ID"]).size()
２４．joblibはpickleよりも大きなメモリに対して計算が速いらしい．
２５．ファイル操作はosを使う．os.path.join
２６．DataFrameのデータ選択は，ixはインデックス，ilocは行番号を指定する．
２７．コマンドをpython側から実行する．
cmd = "python ./preprocess.py"
os.system(cmd)
２８．行もちに変換する（pivotの逆バージョン）
snslocation_train = pd.melt(snslocation_train, id_vars=["date"], var_name= "location", value_name="snslocation")
２９．何故かnp.logだとエラーが出るので，math.logで対数をとる
３０．[0] * 3　という書き方をすると，おかしな現象が起こることがあるから，やらない．

[~([true,false])]   否定形
df[“A”].clip(0.8) 最大８最小０
sklearn.preprocessing import LabelBinarizer
Series.to_list() Seriesをlistに変換
pd.merge(left1,left2, left_on=“key”,right_index=True)  インデックスをmerge
concat([s1,s2,s3])
np.hstack((a,b,c)) concatのnumpyバージョン
np.array(x, dtype=np.float32) numpy型の配列作成
A.sort([“”,””],ascending=[True,True],inplace=True)
result.value_counts()
A[“”].notnull
tb,fb = [],[]
A = pd.DataFrame([])
A.append(B, ignore_index=True)
hstack 水平に配列を結合する
A.as_matrix()　データフレームを行列に変換する
filter_idx = np.one(user_coupons.shape[0], dtyoe=np.bool) True/Falseリストの作成
date.isoweekday()・date.isocalendar()でユニークな数字で返ってくる．
astype() 型を更新する
np.column_stack　列方向にデータを結合する
df.fillna({“columnA”:0.5, “columnsB”:1}, inplace=True)
pd.get_dummies(df[“key”], prefix=“key")
data.str.contains(“gmail")
fig, axis = plt.subplot(2, 1)
A.groupby([“sex], as_index=False).mean()
import pandas.tseries.offsets as offsets 一般的な日付の加減算
# TODO コメントを書く

ix，iloc，loc

使える引数の形式

ix

iloc

loc

名前 (もしくは名前のリスト)

○

-

○

順序 (番号)　(もしくは番号のリスト)

○

○

-

index, もしくは columns と同じ長さの boolのリスト

○

○

○

continuous = list(set(self.features) - set(categoricals)) + ["activity"]
self.combined = pd.concat([self.combined[continuous], combined_categoricals], axis=1)
log2 = lambda x: log(x)/log(2)
os.getcwd()
totalscores = dict([(row[0],0) for row in rows])
DataFrameの概要表示：rossmann_df.info()
