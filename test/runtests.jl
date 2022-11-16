using LOFwithCluster
using Test
using StatsBase

@testset "LOFwithCluster.jl" begin
    # Write your tests here.
    D = 256
    data = [rand(Float32, D) for i in 1:1000]
    lofc = SLOFC(data, cutScore=1.2)
    clsses = getClasses(lofc)
    @show length(clsses)
    @show length(vcat(values(clsses)...))
    @show getIsolate(lofc)
    newpoint = rand(Float32, D)
    isIsolate, clsid = LOFwithCluster.insert!(lofc, newpoint)
    @show isIsolate
    @show clsid
    @show min(lofc.scores...), max(lofc.scores...), StatsBase.middle(lofc.scores), count(lofc.scores .< 1)/length(lofc.scores)

    @time lofc = SLOFC(data)
end
