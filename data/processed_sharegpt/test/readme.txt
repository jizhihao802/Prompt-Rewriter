test：
    test.parquet:第一版数据
    test2.parquet:第二版数据，共256条，将不进行修改放在后面。后缀为"</instruction>。<method>1.调整指令结构 2.缩短指令长度 3.增加任务说明 4.无需进行优化</method>只输出你选择的优化方法的编号，如果选择多种方法则将每个编号用“/”分隔，不输出其他任何内容。"
    test2_half.parquet:第二版数据的一半
    test2_same.parquet:256条相同数据
    test2_shuffle.parquet:第二版数据打乱方法顺序,4(不进行修改）固定的版本
    test2_shuffle2.parquet:第二版数据完全打乱方法顺序的版本
    test2_shuffle3.parquet:第二版训练数据train2_shuffle2.parquet（256-512）的部分，来进行二次测试
    test3.parquet:第二版数据调整方法顺序，用于测试模型选择方法的能力是否与顺序无关。后缀为"</instruction>。<method>1.增加任务说明 2.调整指令结构 3.缩短指令长度 4.无需进行优化</method>只输出你选择的优化方法的编号，如果选择多种方法则将每个编号用“/”分隔，不输出其他任何内容。"
    test4.parquet:测试后缀为"</instruction>。<method>1.增加任务说明 2.调整指令结构 3.缩短指令长度 4.无需进行优化</method>输出你选择的优化方法的编号，如果选择多种方法则将每个编号用“/”分隔，并解释你选择的原因。"
    t1.parquet:测试集test2_shuffle2.parquet遍历所有方法组合生成的oracle文件
    test2_shuffle2_simple.parquet:与t1.parquet相同