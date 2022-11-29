using OptImpSampling
import OptImpSampling: isokann, Isokann, run

@testitem "Isokann()" begin
    import OptImpSampling: Isokann, run
    run(Isokann())
end

@testitem "isokann()" begin
    import OptImpSampling: isokann
    isokann()
end
