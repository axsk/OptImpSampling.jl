using Pkg, Coverage
Pkg.test("OptImpSampling"; coverage=true)
LCOV.writefile("lcov.info", process_folder())
Coverage.clean_folder("src")
