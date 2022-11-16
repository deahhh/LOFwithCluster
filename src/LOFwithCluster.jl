module LOFwithCluster
using HNSW
using Distances
using StatsBase
"""
SLOFC is the main structure for LOFwithCluster. It store points in hnsw, which provides knn-search rapidly.
The algorithom is not the original one.

   k : parameter for `k-nn`
   cutScore : threshold for judge isolate point
   hnsw : data store, and knnseracher
   cls : map from point index to classid
   scores : isolate score for each point, in order. score > 1, means the density of this point is less its k nearest neigbors's.
"""
mutable struct SLOFC
    k::UInt8
    cutScore::Float16
    hnsw::HNSW.HierarchicalNSW
    cls::Dict{UInt32, UInt16}
    scores::Vector{Float16}
    lrdkp::Vector{Float16}
    kdist::Vector{Float16}
end
"""
Constructer for SLOFC.
    length(data) should > 50, for sure. Too less data leads to low accuracy.
"""
function SLOFC(data::Vector{Vector{T}}where T<:Real; k=31, cutScore=1.6, kwargs...)
    @assert length(data)>50
    d = SLOFC(
        convert(UInt8, k),
        convert(Float16, cutScore),
        HNSW.HierarchicalNSW(data; kwargs...),
        Dict{UInt32, UInt16}(),
        Vector{Float16}(),
        Vector{Float16}(),
        Vector{Float16}(),
    )
    add_to_graph!(d.hnsw)

    for (i, p) in enumerate(d.hnsw.data)
        push!(d.kdist, Float16(kdist(d, i|>UInt32)))
    end
    for (i, p) in enumerate(d.hnsw.data)
        push!(d.lrdkp, lrdkp(d, p)|>Float16)
    end

    for (i,p) in enumerate(d.hnsw.data)
        point_analysis!(d, i |> UInt32)
    end
    if length(d.cls) != length(d.hnsw.data)
        # @warn "There are $(length(d.hnsw.data)-length(d.cls)) points been not classified, and will not to be."
    end
    d
end
"""
core algorithom. will be called automatice after insert new data.
return 
    isIsolate : score > cutScore
    clsid : 0 if not sure now. else id.
"""
function point_analysis!(d::SLOFC, n::UInt32)
    score, ids, dist = calcu_lofkp(d, n)
    push!(d.scores, score |> Float16)
    if score < Float16(1.05)
        clst = ids[1:(d.k ÷ 2)]
        bclsf = map(x->haskey(d.cls,x), clst)
        cls_ = map(x->d.cls[x], clst[bclsf])
        if count(bclsf)==1
            clsid = d.cls[clst[findfirst(bclsf)]]
            merge!(d.cls, Dict([id=>clsid for id in clst]))
        elseif count(bclsf)>1
            # @warn "There is at leaset one point with armbigious classid, cutted."
            indx = findfirst(bclsf)
            clsid = d.cls[clst[indx]]
            merge!(d.cls, Dict([id=>clsid for id in clst[1:indx]]))
        else
            clsid = length(d.cls)+1
            merge!(d.cls, Dict([id=>clsid for id in clst]))
        end
    else
        clsid = 0
    end
    
    score > d.cutScore, clsid # isolate or not, class identity
end

function searchKNN(slofc::SLOFC, point::Vector{T}where T<:Real)
    queries = knn_search(slofc.hnsw, [point], slofc.k)
    queries[1][1], queries[2][1]
end

searchKNN(slofc::SLOFC, points::Vector{Vector{T}}where T<:Real) = knn_search(slofc.hnsw, points, slofc.k)

function kdist(slofc::SLOFC, idx::UInt32)
    if length(slofc.kdist) >= idx
        return slofc.kdist[idx]
    else
        point = slofc.hnsw.data[idx]
        ids, dst = searchKNN(slofc, point)
        k_dist, k_indc = findmax(dst)
        k_dist
    end
end

function reachdist(slofc::SLOFC, p::Vector{T}, o::Vector{T})where T<:Real
    max(slofc.hnsw.metric(p, o), kdist(slofc, o))
end
function reachdist(slofc::SLOFC, np::UInt32, no::UInt32)
    @assert np <= length(slofc.hnsw.data) >= no
    max(slofc.hnsw.metric(slofc.hnsw.data[np], slofc.hnsw.data[no]), slofc.kdist[no])
end

function lrdkp(slofc::SLOFC, p::Vector{T} )where T<:Real
    ids_nb, dists = searchKNN(slofc, p)
    slofc.k / max(eps(T), mapreduce(max, +, slofc.kdist[ids_nb], dists))
    # 1 / max(StatsBase.mean(map(x->reachdist(slofc, p, slofc.hnsw.data[x]), searchKNN(slofc, p)[1])), eps(T))
end

function lrdkp(slofc::SLOFC, n::UInt32)
    lrdkp(slofc, slofc.hnsw.data[n])
end
"""
Calculate the `local outlier factor` for point given in SLOF.
"""
lofkp(slofc::SLOFC, p::Vector{T} where T<:Real) = 
    StatsBase.mean(map(x->lrdkp(slofc, slofc.hnsw.data[x]), searchKNN(slofc,p)[1])) / lrdkp(slofc, p)

function calcu_lofkp(slofc::SLOFC, n::UInt32)
    p = slofc.hnsw.data[n]
    ids, dst = searchKNN(slofc, p)
    score = StatsBase.mean(slofc.lrdkp[ids]) / slofc.lrdkp[n]
    score, ids, dst
end

function insert!(slofc::SLOFC, p::Vector{T}where T<:Real)
    HNSW.add!(slofc.hnsw, [p])
    n = UInt32(length(slofc.hnsw.data))
    push!(slofc.kdist, Float16(kdist(slofc, n)))
    push!(slofc.lrdkp, lrdkp(slofc, p))
    point_analysis!(slofc, n)
    # return isolate or not and clsid
end

function getClasses(slofc::SLOFC)
    cls = Dict{UInt16, Vector{UInt32}}()
    for (idx, clsid) in slofc.cls
        if haskey(cls, clsid)
            push!(cls[clsid], idx)
        else
            cls[clsid] = [idx]
        end
    end
    return cls
end

getIsolate(slofc::SLOFC) = findall(x->x>slofc.cutScore, slofc.scores)

function proposed_danger_distance(slofc::SLOFC)
    isos = getIsolate(slofc)
    StatsBase.middle([knn_search(slofc.hnsw, [slofc.hnsw.data[x]], 1)[2][1] for x in isos])
end

export SLOFC, getClasses, getIsolate, insert!, proposed_danger_distance

end