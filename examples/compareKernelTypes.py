#!/usr/bin/env python

# This file is part of ip_diffim.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import unittest
import lsst.utils.tests as tests

import lsst.utils
import lsst.afw.display as afwDisplay
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.utils.logging as logUtils
import lsst.pex.config as pexConfig
import lsst.ip.diffim as ipDiffim


verbosity = 4
logUtils.trace_set_at("lsst.ip.diffim", verbosity)

display = True
writefits = False

fitForBackground = False
constantVarianceWeighting = True

# This one compares DeltaFunction and AlardLupton kernels
defSciencePath = None
defTemplatePath = None


class DiffimTestCases(unittest.TestCase):
    # D = I - (K.x.T + bg)

    def setUp(self):
        self.configAL = ipDiffim.ImagePsfMatchTask.ConfigClass()
        self.configAL.kernel.name = "AL"
        self.subconfigAL = self.configAL.kernel.active

        self.configDF = ipDiffim.ImagePsfMatchTask.ConfigClass()
        self.configDF.kernel.name = "DF"
        self.subconfigDF = self.configDF.kernel.active

        self.configDFr = ipDiffim.ImagePsfMatchTask.ConfigClass()
        self.configDFr.kernel.name = "DF"
        self.subconfigDFr = self.configDFr.kernel.active

        self.subconfigDF.useRegularization = False
        self.subconfigDFr.useRegularization = True
        self.subconfigDFr.lambdaValue = 1000.0

        self.subconfigAL.fitForBackground = fitForBackground
        self.subconfigDF.fitForBackground = fitForBackground
        self.subconfigDFr.fitForBackground = fitForBackground

        self.subconfigAL.constantVarianceWeighting = constantVarianceWeighting
        self.subconfigDF.constantVarianceWeighting = constantVarianceWeighting
        self.subconfigDFr.constantVarianceWeighting = constantVarianceWeighting

        self.kListAL = ipDiffim.makeKernelBasisList(self.subconfigAL)
        self.kListDF = ipDiffim.makeKernelBasisList(self.subconfigDF)
        self.kListDFr = ipDiffim.makeKernelBasisList(self.subconfigDFr)
        self.hMatDFr = ipDiffim.makeRegularizationMatrix(pexConfig.makePropertySet(self.subconfigDFr))

        self.bskvAL = ipDiffim.BuildSingleKernelVisitorF(self.kListAL,
                                                         pexConfig.makePropertySet(self.subconfigAL))
        self.bskvDF = ipDiffim.BuildSingleKernelVisitorF(self.kListDF,
                                                         pexConfig.makePropertySet(self.subconfigDF))
        self.bskvDFr = ipDiffim.BuildSingleKernelVisitorF(self.kListDFr,
                                                          pexConfig.makePropertySet(self.subconfigDF),
                                                          self.hMatDFr)

        defSciencePath = globals()['defSciencePath']
        defTemplatePath = globals()['defTemplatePath']
        if defSciencePath and defTemplatePath:
            self.scienceExposure = afwImage.ExposureF(defSciencePath)
            self.templateExposure = afwImage.ExposureF(defTemplatePath)
        else:
            defDataDir = lsst.utils.getPackageDir('afwdata')
            defSciencePath = os.path.join(defDataDir, "DC3a-Sim", "sci", "v26-e0",
                                          "v26-e0-c011-a00.sci.fits")
            defTemplatePath = os.path.join(defDataDir, "DC3a-Sim", "sci", "v5-e0",
                                           "v5-e0-c011-a00.sci.fits")

            self.scienceExposure = afwImage.ExposureF(defSciencePath)
            self.templateExposure = afwImage.ExposureF(defTemplatePath)
            warper = afwMath.Warper.fromConfig(self.subconfigAL.warpingConfig)
            self.templateExposure = warper.warpExposure(self.scienceExposure.getWcs(), self.templateExposure,
                                                        destBBox=self.scienceExposure.getBBox())

        #
        tmi = self.templateExposure.getMaskedImage()
        smi = self.scienceExposure.getMaskedImage()

        # Object detection
        detConfig = self.subconfigAL.detectionConfig
        detps = pexConfig.makePropertySet(detConfig)
        detps["detThreshold"] = 50.
        detps["detThresholdType"] = "stdev"
        detps["detOnTemplate"] = False
        kcDetect = ipDiffim.KernelCandidateDetectionF(detps)
        kcDetect.apply(tmi, smi)
        self.footprints = kcDetect.getFootprints()

    def tearDown(self):
        del self.kListAL
        del self.kListDF
        del self.kListDFr
        del self.hMatDFr
        del self.bskvAL
        del self.bskvDF
        del self.bskvDFr
        del self.scienceExposure
        del self.templateExposure
        del self.footprints

    def apply(self, ps, visitor, xloc, yloc, tmi, smi):
        # image statistics
        dStats = ipDiffim.ImageStatisticsF(ps)
        kc = ipDiffim.makeKernelCandidate(xloc, yloc, tmi, smi, ps)
        visitor.processCandidate(kc)
        kim = kc.getKernelImage(ipDiffim.KernelCandidateF.RECENT)
        diffIm = kc.getDifferenceImage(ipDiffim.KernelCandidateF.RECENT)
        kSum = kc.getKsum(ipDiffim.KernelCandidateF.RECENT)
        bg = kc.getBackground(ipDiffim.KernelCandidateF.RECENT)

        bbox = kc.getKernel(ipDiffim.KernelCandidateF.RECENT).shrinkBBox(diffIm.getBBox(afwImage.LOCAL))
        diffIm = afwImage.MaskedImageF(diffIm, bbox, origin=afwImage.LOCAL)
        dStats.apply(diffIm)

        dmean = afwMath.makeStatistics(diffIm.getImage(), afwMath.MEAN).getValue()
        dstd = afwMath.makeStatistics(diffIm.getImage(), afwMath.STDEV).getValue()
        vmean = afwMath.makeStatistics(diffIm.getVariance(), afwMath.MEAN).getValue()
        return kSum, bg, dmean, dstd, vmean, kim, diffIm, kc, dStats

    def applyVisitor(self, invert=False, xloc=397, yloc=580):
        print('# %.2f %.2f' % (xloc, yloc))

        imsize = int(3 * self.subconfigAL.kernelSize)

        # chop out a region around a known object
        bbox = afwGeom.Box2I(afwGeom.Point2I(xloc - imsize//2,
                                             yloc - imsize//2),
                             afwGeom.Point2I(xloc + imsize//2,
                                             yloc + imsize//2))

        # sometimes the box goes off the image; no big deal...
        try:
            if invert:
                tmi = afwImage.MaskedImageF(self.scienceExposure.getMaskedImage(), bbox,
                                            origin=afwImage.LOCAL)
                smi = afwImage.MaskedImageF(self.templateExposure.getMaskedImage(), bbox,
                                            orign=afwImage.LOCAL)
            else:
                smi = afwImage.MaskedImageF(self.scienceExposure.getMaskedImage(), bbox,
                                            origin=afwImage.LOCAL)
                tmi = afwImage.MaskedImageF(self.templateExposure.getMaskedImage(), bbox,
                                            origin=afwImage.LOCAL)
        except Exception:
            return None

        #
        ###
        #

        # delta function kernel
        resultsDF = self.apply(pexConfig.makePropertySet(self.subconfigDF), self.bskvDF, xloc, yloc, tmi, smi)
        kSumDF, bgDF, dmeanDF, dstdDF, vmeanDF, kImageOutDF, diffImDF, kcDF, dStats = resultsDF
        kcDF.getKernelSolution(ipDiffim.KernelCandidateF.RECENT).getConditionNumber(
            ipDiffim.KernelSolution.EIGENVALUE)
        kcDF.getKernelSolution(ipDiffim.KernelCandidateF.RECENT).getConditionNumber(
            ipDiffim.KernelSolution.SVD)
        print('DF Diffim residuals : %.2f +/- %.2f; %.2f, %.2f; %.2f %.2f, %.2f' % (dStats.getMean(),
                                                                                    dStats.getRms(),
                                                                                    kSumDF, bgDF,
                                                                                    dmeanDF, dstdDF, vmeanDF))
        if display:
            afwDisplay.Display(frame=1).mtv(tmi)  # ds9 switches frame 0 and 1 for some reason
            afwDisplay.Display(frame=0).mtv(smi)
            afwDisplay.Display(frame=2).mtv(kImageOutDF)
            afwDisplay.Display(frame=3).mtv(diffImDF)
        if writefits:
            tmi.writeFits('t')
            smi.writeFits('s')
            kImageOutDF.writeFits('kDF.fits')
            diffImDF.writeFits('dDF')

        #
        ###
        #

        # regularized delta function kernel
        resultsDFr = self.apply(pexConfig.makePropertySet(self.subconfigDFr),
                                self.bskvDFr, xloc, yloc, tmi, smi)
        kSumDFr, bgDFr, dmeanDFr, dstdDFr, vmeanDFr, kImageOutDFr, diffImDFr, kcDFr, dStats = resultsDFr
        kcDFr.getKernelSolution(ipDiffim.KernelCandidateF.RECENT).getConditionNumber(
            ipDiffim.KernelSolution.EIGENVALUE)
        kcDFr.getKernelSolution(ipDiffim.KernelCandidateF.RECENT).getConditionNumber(
            ipDiffim.KernelSolution.SVD)
        print('DFr Diffim residuals : %.2f +/- %.2f; %.2f, %.2f; %.2f %.2f, %.2f' %
              (dStats.getMean(), dStats.getRms(), kSumDFr, bgDFr, dmeanDFr, dstdDFr, vmeanDFr))
        if display:
            afwDisplay.Display(frame=4).mtv(tmi)
            afwDisplay.Display(frame=5).mtv(smi)
            afwDisplay.Display(frame=6).mtv(kImageOutDFr)
            afwDisplay.Display(frame=7).mtv(diffImDFr)
        if writefits:
            kImageOutDFr.writeFits('kDFr.fits')
            diffImDFr.writeFits('dDFr')

        #
        ###
        #

        # alard-lupton kernel
        resultsAL = self.apply(pexConfig.makePropertySet(self.subconfigAL), self.bskvAL, xloc, yloc, tmi, smi)
        kSumAL, bgAL, dmeanAL, dstdAL, vmeanAL, kImageOutAL, diffImAL, kcAL, dStats = resultsAL
        kcAL.getKernelSolution(ipDiffim.KernelCandidateF.RECENT).getConditionNumber(
            ipDiffim.KernelSolution.EIGENVALUE)
        kcAL.getKernelSolution(ipDiffim.KernelCandidateF.RECENT).getConditionNumber(
            ipDiffim.KernelSolution.SVD)
        print('AL Diffim residuals : %.2f +/- %.2f; %.2f, %.2f; %.2f %.2f, %.2f' % (dStats.getMean(),
                                                                                    dStats.getRms(),
                                                                                    kSumAL, bgAL,
                                                                                    dmeanAL, dstdAL, vmeanAL))
        # outputs
        if display:
            afwDisplay.Display(frame=8).mtv(tmi)
            afwDisplay.Display(frame=9).mtv(smi)
            afwDisplay.Display(frame=10).mtv(kImageOutAL)
            afwDisplay.Display(frame=11).mtv(diffImAL)
        if writefits:
            kImageOutAL.writeFits('kAL.fits')
            diffImAL.writeFits('dAL')

        input('Next: ')

    def testFunctor(self):
        for fp in self.footprints:
            # note this returns the kernel images
            self.applyVisitor(invert=False,
                              xloc=int(0.5 * (fp.getBBox().getMinX() + fp.getBBox().getMaxX())),
                              yloc=int(0.5 * (fp.getBBox().getMinY() + fp.getBBox().getMaxY())))

#####


def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(DiffimTestCases)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)


def run(doExit=False):
    """Run the tests"""
    tests.run(suite(), doExit)


if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-n', dest='nodisplay', action='store_true', default=False)
    parser.add_option('-w', dest='writefits', action='store_true', default=False)
    parser.add_option('-t', dest='defTemplatePath')
    parser.add_option('-i', dest='defSciencePath')
    (opt, args) = parser.parse_args()

    display = not opt.nodisplay
    writefits = opt.writefits
    if opt.defTemplatePath and opt.defSciencePath:
        defTemplatePath = opt.defTemplatePath
        defSciencePath = opt.defSciencePath

    run(True)
