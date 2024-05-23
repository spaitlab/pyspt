import ruptures as rpt

def findchangepts(signal,n_bkps):
    # 使用Binseg方法进行变化点检测
    mode = "l2"  # 使用L2损失，也称为最小均方误差
    algo = rpt.Binseg(model=mode).fit(signal)
    my_bkps = algo.predict(n_bkps=n_bkps-1)
    return my_bkps



