using OptImpSampling: mean_and_std, ProblemOptSqra, Isokann, run, train, LogVarProblem

function testall()
    @time mean_and_std(ProblemOptSqra(), [0.], 2)
    @time run(Isokann(poweriter=1, learniter=1))
    @time train(LogVarProblem(),1,2)
end

testall()

import OptImpSampling
# control
OptImpSampling.test_ControlledSDE()
OptImpSampling.test_optcontrol()
OptImpSampling.test_compare_controls()

using OptImpSampling: isokann, Doublewell

isokann(Doublewell())
#isokann(Doublewell(dim=2))  # TODO: this fails because of the fixed SVector dimension
