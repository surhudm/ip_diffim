import unittest


import lsst.utils.tests
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom as geom
import lsst.ip.diffim as ipDiffim
import lsst.utils.logging as logUtils
import lsst.pex.config as pexConfig

logUtils.trace_set_at("lsst.ip.diffim", 4)


class DiffimTestCases(unittest.TestCase):

    def setUp(self):
        self.config = ipDiffim.ImagePsfMatchTask.ConfigClass()
        self.config.kernel.name = "DF"
        self.subconfig = self.config.kernel.active

        self.ps = pexConfig.makePropertySet(self.subconfig)

        self.ps["useRegularization"] = False
        self.ps["checkConditionNumber"] = False  # I am making shady kernels by hand
        self.ps["useCoreStats"] = False  # I am making off-center resids
        self.kList = ipDiffim.makeKernelBasisList(self.subconfig)
        self.size = 51

    def makeCandidate(self, kSum, x, y):
        mi1 = afwImage.MaskedImageF(geom.Extent2I(self.size, self.size))
        mi1.getVariance().set(1.0)  # avoid NaNs
        mi1[self.size//2, self.size//2, afwImage.LOCAL] = (1, 0x0, 1)
        mi2 = afwImage.MaskedImageF(geom.Extent2I(self.size, self.size))
        mi2.getVariance().set(1.0)  # avoid NaNs
        mi2[self.size//2, self.size//2, afwImage.LOCAL] = (kSum, 0x0, kSum)
        kc = ipDiffim.makeKernelCandidate(x, y, mi1, mi2, self.ps)

        return kc

    def testWithOneBasis(self):
        self.runWithOneBasis(False)
        self.runWithOneBasis(True)

    def runWithOneBasis(self, useRegularization):
        kc1 = self.makeCandidate(1, 0.0, 0.0)
        kc2 = self.makeCandidate(2, 0.0, 0.0)
        kc3 = self.makeCandidate(3, 0.0, 0.0)

        if useRegularization:
            hMat = ipDiffim.makeRegularizationMatrix(self.ps)
            bskv = ipDiffim.BuildSingleKernelVisitorF(self.kList, self.ps, hMat)
        else:
            bskv = ipDiffim.BuildSingleKernelVisitorF(self.kList, self.ps)

        bskv.processCandidate(kc1)
        bskv.processCandidate(kc2)
        bskv.processCandidate(kc3)

        # Initialized
        self.assertEqual(kc1.isInitialized(), True)
        self.assertEqual(kc2.isInitialized(), True)
        self.assertEqual(kc3.isInitialized(), True)

        # Is a solution
        try:
            kc1.getKernelSolution(ipDiffim.KernelCandidateF.RECENT)
            kc2.getKernelSolution(ipDiffim.KernelCandidateF.RECENT)
            kc3.getKernelSolution(ipDiffim.KernelCandidateF.RECENT)
        except Exception as e:
            print(e)
            self.fail()

        # Its not the Pca one
        try:
            kc1.getKernelSolution(ipDiffim.KernelCandidateF.PCA)
            kc2.getKernelSolution(ipDiffim.KernelCandidateF.PCA)
            kc3.getKernelSolution(ipDiffim.KernelCandidateF.PCA)
        except Exception:
            pass
        else:
            self.fail()

        # Processed all of them
        self.assertEqual(bskv.getNProcessed(), 3)

        # Rejected none
        self.assertEqual(bskv.getNRejected(), 0)

        # Skips built candidates
        bskv.reset()
        bskv.setSkipBuilt(True)
        bskv.processCandidate(kc1)
        bskv.processCandidate(kc2)
        bskv.processCandidate(kc3)
        # Processed none of them
        self.assertEqual(bskv.getNProcessed(), 0)

    def testWithThreeBases(self):
        kc1 = self.makeCandidate(1, 0.0, 0.0)
        kc2 = self.makeCandidate(2, 0.0, 0.0)
        kc3 = self.makeCandidate(3, 0.0, 0.0)
        bskv1 = ipDiffim.BuildSingleKernelVisitorF(self.kList, self.ps)
        bskv1.processCandidate(kc1)
        bskv1.processCandidate(kc2)
        bskv1.processCandidate(kc3)
        self.assertEqual(bskv1.getNProcessed(), 3)

        # make sure orig solution is the current one
        soln1_1 = kc1.getKernelSolution(ipDiffim.KernelCandidateF.ORIG).getId()
        soln2_1 = kc2.getKernelSolution(ipDiffim.KernelCandidateF.ORIG).getId()
        soln3_1 = kc3.getKernelSolution(ipDiffim.KernelCandidateF.ORIG).getId()
        self.assertEqual(soln1_1, kc1.getKernelSolution(ipDiffim.KernelCandidateF.RECENT).getId())
        self.assertEqual(soln2_1, kc2.getKernelSolution(ipDiffim.KernelCandidateF.RECENT).getId())
        self.assertEqual(soln3_1, kc3.getKernelSolution(ipDiffim.KernelCandidateF.RECENT).getId())

        # do pca basis; visit manually since visitCandidates is still broken
        imagePca = ipDiffim.KernelPcaD()
        kpv = ipDiffim.KernelPcaVisitorF(imagePca)
        kpv.processCandidate(kc1)
        kpv.processCandidate(kc2)
        kpv.processCandidate(kc3)
        kpv.subtractMean()
        imagePca.analyze()
        eigenKernels = []
        eigenKernels.append(kpv.getEigenKernels()[0])
        self.assertEqual(len(eigenKernels), 1)  # the other eKernels are 0.0 and you can't get their coeffs!

        # do twice to mimic a Pca loop
        bskv2 = ipDiffim.BuildSingleKernelVisitorF(eigenKernels, self.ps)
        bskv2.setSkipBuilt(False)
        bskv2.processCandidate(kc1)
        bskv2.processCandidate(kc2)
        bskv2.processCandidate(kc3)
        self.assertEqual(bskv2.getNProcessed(), 3)

        soln1_2 = kc1.getKernelSolution(ipDiffim.KernelCandidateF.PCA).getId()
        soln2_2 = kc2.getKernelSolution(ipDiffim.KernelCandidateF.PCA).getId()
        soln3_2 = kc3.getKernelSolution(ipDiffim.KernelCandidateF.PCA).getId()
        # pca is recent
        self.assertEqual(soln1_2, kc1.getKernelSolution(ipDiffim.KernelCandidateF.RECENT).getId())
        self.assertEqual(soln2_2, kc2.getKernelSolution(ipDiffim.KernelCandidateF.RECENT).getId())
        self.assertEqual(soln3_2, kc3.getKernelSolution(ipDiffim.KernelCandidateF.RECENT).getId())
        # orig is still orig
        self.assertEqual(soln1_1, kc1.getKernelSolution(ipDiffim.KernelCandidateF.ORIG).getId())
        self.assertEqual(soln2_1, kc2.getKernelSolution(ipDiffim.KernelCandidateF.ORIG).getId())
        self.assertEqual(soln3_1, kc3.getKernelSolution(ipDiffim.KernelCandidateF.ORIG).getId())
        # pca is not orig
        self.assertNotEqual(soln1_2, soln1_1)
        self.assertNotEqual(soln2_2, soln2_1)
        self.assertNotEqual(soln3_2, soln3_1)

        # do twice to mimic a Pca loop
        bskv3 = ipDiffim.BuildSingleKernelVisitorF(eigenKernels, self.ps)
        bskv3.setSkipBuilt(False)
        bskv3.processCandidate(kc1)
        bskv3.processCandidate(kc2)
        bskv3.processCandidate(kc3)
        self.assertEqual(bskv3.getNProcessed(), 3)

        soln1_3 = kc1.getKernelSolution(ipDiffim.KernelCandidateF.PCA).getId()
        soln2_3 = kc2.getKernelSolution(ipDiffim.KernelCandidateF.PCA).getId()
        soln3_3 = kc3.getKernelSolution(ipDiffim.KernelCandidateF.PCA).getId()
        # pca is recent
        self.assertEqual(soln1_3, kc1.getKernelSolution(ipDiffim.KernelCandidateF.RECENT).getId())
        self.assertEqual(soln2_3, kc2.getKernelSolution(ipDiffim.KernelCandidateF.RECENT).getId())
        self.assertEqual(soln3_3, kc3.getKernelSolution(ipDiffim.KernelCandidateF.RECENT).getId())
        # pca is not previous pca
        self.assertNotEqual(soln1_2, soln1_3)
        self.assertNotEqual(soln2_2, soln2_3)
        self.assertNotEqual(soln3_2, soln3_3)

    def testRejection(self):
        # we need to construct a candidate whose shape does not
        # match the underlying basis
        #
        # so lets just make a kernel list with all the power in
        # the center, but the candidate requires some off center
        # power
        kc1 = self.makeCandidate(1, 0.0, 0.0)
        kc2 = self.makeCandidate(2, 0.0, 0.0)
        kc3 = self.makeCandidate(3, 0.0, 0.0)
        bskv1 = ipDiffim.BuildSingleKernelVisitorF(self.kList, self.ps)
        bskv1.processCandidate(kc1)
        bskv1.processCandidate(kc2)
        bskv1.processCandidate(kc3)

        imagePca = ipDiffim.KernelPcaD()
        kpv = ipDiffim.KernelPcaVisitorF(imagePca)
        kpv.processCandidate(kc1)
        kpv.processCandidate(kc2)
        kpv.processCandidate(kc3)
        kpv.subtractMean()
        imagePca.analyze()
        eigenKernels = []
        eigenKernels.append(kpv.getEigenKernels()[0])
        self.assertEqual(len(eigenKernels), 1)

        # bogus candidate
        mi1 = afwImage.MaskedImageF(geom.Extent2I(self.size, self.size))
        mi1.getVariance().set(0.1)
        mi1[self.size//2, self.size//2, afwImage.LOCAL] = (1, 0x0, 1)
        mi2 = afwImage.MaskedImageF(geom.Extent2I(self.size, self.size))
        mi2.getVariance().set(0.1)
        # make it high enough to make the mean resids large
        mi2[self.size//3, self.size//3, afwImage.LOCAL] = (self.size**2, 0x0, 1)
        kc4 = ipDiffim.makeKernelCandidate(0, 0, mi1, mi2, self.ps)
        self.assertEqual(kc4.getStatus(), afwMath.SpatialCellCandidate.UNKNOWN)

        # process with eigenKernels
        bskv2 = ipDiffim.BuildSingleKernelVisitorF(eigenKernels, self.ps)
        bskv2.setSkipBuilt(False)
        bskv2.processCandidate(kc1)
        bskv2.processCandidate(kc2)
        bskv2.processCandidate(kc3)
        bskv2.processCandidate(kc4)

        self.assertEqual(bskv2.getNProcessed(), 4)
        self.assertEqual(bskv2.getNRejected(), 1)

        self.assertEqual(kc1.getStatus(), afwMath.SpatialCellCandidate.GOOD)
        self.assertEqual(kc2.getStatus(), afwMath.SpatialCellCandidate.GOOD)
        self.assertEqual(kc3.getStatus(), afwMath.SpatialCellCandidate.GOOD)
        self.assertEqual(kc4.getStatus(), afwMath.SpatialCellCandidate.BAD)

    def testVisit(self, nCell=3):
        bskv = ipDiffim.BuildSingleKernelVisitorF(self.kList, self.ps)

        sizeCellX = self.ps["sizeCellX"]
        sizeCellY = self.ps["sizeCellY"]

        kernelCellSet = afwMath.SpatialCellSet(geom.Box2I(geom.Point2I(0, 0),
                                                          geom.Extent2I(sizeCellX * nCell,
                                                                        sizeCellY * nCell)),
                                               sizeCellX,
                                               sizeCellY)
        nTot = 0
        for candX in range(nCell):
            for candY in range(nCell):
                if candX == nCell // 2 and candY == nCell // 2:
                    kc = self.makeCandidate(100.0,
                                            candX * sizeCellX + sizeCellX // 2,
                                            candY * sizeCellY + sizeCellY // 2)
                else:
                    kc = self.makeCandidate(1.0,
                                            candX * sizeCellX + sizeCellX // 2,
                                            candY * sizeCellY + sizeCellY // 2)
                kernelCellSet.insertCandidate(kc)
                nTot += 1

        kernelCellSet.visitCandidates(bskv, 1)
        self.assertEqual(bskv.getNProcessed(), nTot)
        self.assertEqual(bskv.getNRejected(), 0)

        for cell in kernelCellSet.getCellList():
            for cand in cell.begin(False):
                self.assertEqual(cand.getStatus(), afwMath.SpatialCellCandidate.GOOD)

    def tearDown(self):
        del self.config
        del self.ps
        del self.kList


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
