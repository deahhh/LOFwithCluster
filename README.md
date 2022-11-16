# LOFwithCluster
LOF means "Local Outlier Factor", a alogrithom for isolate(abnormal) point discovering.

With Claster means it will give cluster for normal points.

# Usage
```Julia
using LOFwithCluster
data = [rand(256) for i in 1:1000]
lofc = SLOFC(data)
@show getClasses(lofc)
@show getIsolate(lofc)
newpoint = rand(256)
isIsolate, clsid = insert!(lofc, newpoint)
@show isIsolate
@show clsid
```