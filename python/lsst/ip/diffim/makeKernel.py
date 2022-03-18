# This file is part of pipe_tasks.
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

__all__ = ["MakeKernelConfig", "MakeKernelTask"]

import numpy as np

import lsst.afw.detection
import lsst.afw.image
import lsst.afw.math
import lsst.afw.table
import lsst.daf.base
from lsst.meas.algorithms import SourceDetectionTask, SubtractBackgroundTask
from lsst.meas.base import SingleFrameMeasurementTask
import lsst.pex.config
import lsst.pipe.base

from .makeKernelBasisList import makeKernelBasisList
from .psfMatch import PsfMatchConfig, PsfMatchTask, PsfMatchConfigAL, PsfMatchConfigDF

from . import diffimLib
from . import diffimTools


class MakeKernelConfig(PsfMatchConfig):
    kernel = lsst.pex.config.ConfigChoiceField(
        doc="kernel type",
        typemap=dict(
            AL=PsfMatchConfigAL,
            DF=PsfMatchConfigDF
        ),
        default="AL",
    )
    selectDetection = lsst.pex.config.ConfigurableField(
        target=SourceDetectionTask,
        doc="Initial detections used to feed stars to kernel fitting",
    )
    selectMeasurement = lsst.pex.config.ConfigurableField(
        target=SingleFrameMeasurementTask,
        doc="Initial measurements used to feed stars to kernel fitting",
    )

    def setDefaults(self):
        # High sigma detections only
        self.selectDetection.reEstimateBackground = False
        self.selectDetection.thresholdValue = 10.0

        # Minimal set of measurments for star selection
        self.selectMeasurement.algorithms.names.clear()
        self.selectMeasurement.algorithms.names = ('base_SdssCentroid', 'base_PsfFlux', 'base_PixelFlags',
                                                   'base_SdssShape', 'base_GaussianFlux', 'base_SkyCoord')
        self.selectMeasurement.slots.modelFlux = None
        self.selectMeasurement.slots.apFlux = None
        self.selectMeasurement.slots.calibFlux = None


class MakeKernelTask(PsfMatchTask):
    ConfigClass = MakeKernelConfig
    _DefaultName = "makeALKernel"

    def __init__(self, *args, **kwargs):
        PsfMatchTask.__init__(self, *args, **kwargs)
        self.kConfig = self.config.kernel.active
        # the background subtraction task uses a config from an unusual location,
        # so cannot easily be constructed with makeSubtask
        self.background = SubtractBackgroundTask(config=self.kConfig.afwBackgroundConfig, name="background",
                                                 parentTask=self)
        self.selectSchema = lsst.afw.table.SourceTable.makeMinimalSchema()
        self.selectAlgMetadata = lsst.daf.base.PropertyList()
        self.makeSubtask("selectDetection", schema=self.selectSchema)
        self.makeSubtask("selectMeasurement", schema=self.selectSchema, algMetadata=self.selectAlgMetadata)

    def run(self, template, science, kernelSources, preconvolved=False):
        kernelCellSet = self._buildCellSet(template.maskedImage, science.maskedImage, kernelSources)
        templateFwhmPix = self._getFwhmPix(template)
        scienceFwhmPix = self._getFwhmPix(science)
        if preconvolved:
            scienceFwhmPix *= np.sqrt(2)
        basisList = self.makeKernelBasisList(templateFwhmPix, scienceFwhmPix,
                                             metadata=self.metadata)
        spatialSolution, psfMatchingKernel, backgroundModel = self._solve(kernelCellSet, basisList)
        return lsst.pipe.base.Struct(
            psfMatchingKernel=psfMatchingKernel,
            backgroundModel=backgroundModel,
            kernelCellSet=kernelCellSet,
        )

    def selectKernelSources(self, template, science, candidateList=None, preconvolved=False):
        templateFwhmPix = self._getFwhmPix(template)
        scienceFwhmPix = self._getFwhmPix(science)
        if preconvolved:
            scienceFwhmPix *= np.sqrt(2)
        kernelSize = self.makeKernelBasisList(templateFwhmPix, scienceFwhmPix)[0].getWidth()
        kernelSources = self.makeCandidateList(template, science, kernelSize,
                                               candidateList=candidateList,
                                               preconvolved=preconvolved)
        return kernelSources

    def getSelectSources(self, exposure, sigma=None, doSmooth=True, idFactory=None):
        """Get sources to use for Psf-matching.

        This method runs detection and measurement on an exposure.
        The returned set of sources will be used as candidates for
        Psf-matching.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure on which to run detection/measurement
        sigma : `float`
            Detection threshold
        doSmooth : `bool`
            Whether or not to smooth the Exposure with Psf before detection
        idFactory :
            Factory for the generation of Source ids

        Returns
        -------
        selectSources :
            source catalog containing candidates for the Psf-matching
        """
        if idFactory:
            table = lsst.afw.table.SourceTable.make(self.selectSchema, idFactory)
        else:
            table = lsst.afw.table.SourceTable.make(self.selectSchema)
        mi = exposure.getMaskedImage()

        imArr = mi.getImage().getArray()
        maskArr = mi.getMask().getArray()
        miArr = np.ma.masked_array(imArr, mask=maskArr)
        try:
            fitBg = self.background.fitBackground(mi)
            bkgd = fitBg.getImageF(self.background.config.algorithm,
                                   self.background.config.undersampleStyle)
        except Exception:
            self.log.warning("Failed to get background model. Falling back to median background estimation")
            bkgd = np.ma.median(miArr)

        # Take off background for detection
        mi -= bkgd
        try:
            table.setMetadata(self.selectAlgMetadata)
            detRet = self.selectDetection.run(
                table=table,
                exposure=exposure,
                sigma=sigma,
                doSmooth=doSmooth
            )
            selectSources = detRet.sources
            self.selectMeasurement.run(measCat=selectSources, exposure=exposure)
        finally:
            # Put back on the background in case it is needed down stream
            mi += bkgd
            del bkgd
        return selectSources

    def makeCandidateList(self, templateExposure, scienceExposure, kernelSize,
                          candidateList=None, preconvolved=False):
        """Make a list of acceptable KernelCandidates.

        Accept or generate a list of candidate sources for
        Psf-matching, and examine the Mask planes in both of the
        images for indications of bad pixels

        Parameters
        ----------
        templateExposure : `lsst.afw.image.Exposure`
            Exposure that will be convolved
        scienceExposure : `lsst.afw.image.Exposure`
            Exposure that will be matched-to
        kernelSize : `float`
            Dimensions of the Psf-matching Kernel, used to grow detection footprints
        candidateList : `list`, optional
            List of Sources to examine. Elements must be of type afw.table.Source
            or a type that wraps a Source and has a getSource() method, such as
            meas.algorithms.PsfCandidateF.
        preconvolved : bool, optional
            Description

        Returns
        -------
        candidateList : `list` of `dict`
            A list of dicts having a "source" and "footprint"
            field for the Sources deemed to be appropriate for Psf
            matching

        Raises
        ------
        RuntimeError
            Description
        """
        if candidateList is None:
            candidateList = self.getSelectSources(scienceExposure, doSmooth=not preconvolved)

        if len(candidateList) < 1:
            raise RuntimeError("No candidates in candidateList")

        listTypes = set(type(x) for x in candidateList)
        if len(listTypes) > 1:
            raise RuntimeError("Candidate list contains mixed types: %s" % [t for t in listTypes])

        if not isinstance(candidateList[0], lsst.afw.table.SourceRecord):
            try:
                candidateList[0].getSource()
            except Exception as e:
                raise RuntimeError(f"Candidate List is of type: {type(candidateList[0])} "
                                   "Can only make candidate list from list of afwTable.SourceRecords, "
                                   f"measAlg.PsfCandidateF or other type with a getSource() method: {e}")
            candidateList = [c.getSource() for c in candidateList]

        candidateList = diffimTools.sourceToFootprintList(candidateList,
                                                          templateExposure, scienceExposure,
                                                          kernelSize,
                                                          self.kConfig.detectionConfig,
                                                          self.log)
        if len(candidateList) == 0:
            raise RuntimeError("Cannot find any objects suitable for KernelCandidacy")

        return candidateList

    def makeKernelBasisList(self, targetFwhmPix=None, referenceFwhmPix=None,
                            basisDegGauss=None, basisSigmaGauss=None, metadata=None):
        """Wrapper to set log messages for
        `lsst.ip.diffim.makeKernelBasisList`.

        Parameters
        ----------
        targetFwhmPix : `float`, optional
            Passed on to `lsst.ip.diffim.generateAlardLuptonBasisList`.
            Not used for delta function basis sets.
        referenceFwhmPix : `float`, optional
            Passed on to `lsst.ip.diffim.generateAlardLuptonBasisList`.
            Not used for delta function basis sets.
        basisDegGauss : `list` of `int`, optional
            Passed on to `lsst.ip.diffim.generateAlardLuptonBasisList`.
            Not used for delta function basis sets.
        basisSigmaGauss : `list` of `int`, optional
            Passed on to `lsst.ip.diffim.generateAlardLuptonBasisList`.
            Not used for delta function basis sets.
        metadata : `lsst.daf.base.PropertySet`, optional
            Passed on to `lsst.ip.diffim.generateAlardLuptonBasisList`.
            Not used for delta function basis sets.

        Returns
        -------
        basisList: `list` of `lsst.afw.math.kernel.FixedKernel`
            List of basis kernels.
        """
        basisList = makeKernelBasisList(self.kConfig,
                                        targetFwhmPix=targetFwhmPix,
                                        referenceFwhmPix=referenceFwhmPix,
                                        basisDegGauss=basisDegGauss,
                                        basisSigmaGauss=basisSigmaGauss,
                                        metadata=metadata)
        if targetFwhmPix == referenceFwhmPix:
            self.log.info("Target and reference psf fwhms are equal, falling back to config values")
        elif referenceFwhmPix > targetFwhmPix:
            self.log.info("Reference psf fwhm is the greater, normal convolution mode")
        else:
            self.log.info("Target psf fwhm is the greater, deconvolution mode")

        return basisList

    def _buildCellSet(self, templateMaskedImage, scienceMaskedImage, candidateList):
        """Build a SpatialCellSet for use with the solve method.

        Parameters
        ----------
        templateMaskedImage : `lsst.afw.image.MaskedImage`
            MaskedImage to PSF-matched to scienceMaskedImage
        scienceMaskedImage : `lsst.afw.image.MaskedImage`
            Reference MaskedImage
        candidateList : `list`
            A list of footprints/maskedImages for kernel candidates;

            - Currently supported: list of Footprints or measAlg.PsfCandidateF

        Returns
        -------
        kernelCellSet : `lsst.afw.math.SpatialCellSet`
            a SpatialCellSet for use with self._solve
        """
        if not candidateList:
            raise RuntimeError("Candidate list must be populated by makeCandidateList")

        sizeCellX, sizeCellY = self._adaptCellSize(candidateList)

        # Object to store the KernelCandidates for spatial modeling
        kernelCellSet = lsst.afw.math.SpatialCellSet(templateMaskedImage.getBBox(),
                                                     sizeCellX, sizeCellY)

        ps = lsst.pex.config.makePropertySet(self.kConfig)
        # Place candidates within the spatial grid
        for cand in candidateList:
            if isinstance(cand, lsst.afw.detection.Footprint):
                bbox = cand.getBBox()
            else:
                bbox = cand['footprint'].getBBox()
            tmi = lsst.afw.image.MaskedImageF(templateMaskedImage, bbox)
            smi = lsst.afw.image.MaskedImageF(scienceMaskedImage, bbox)

            if not isinstance(cand, lsst.afw.detection.Footprint):
                if 'source' in cand:
                    cand = cand['source']
            xPos = cand.getCentroid()[0]
            yPos = cand.getCentroid()[1]
            cand = diffimLib.makeKernelCandidate(xPos, yPos, tmi, smi, ps)

            self.log.debug("Candidate %d at %f, %f", cand.getId(), cand.getXCenter(), cand.getYCenter())
            kernelCellSet.insertCandidate(cand)

        return kernelCellSet

    def _adaptCellSize(self, candidateList):
        """NOT IMPLEMENTED YET.
        """
        return self.kConfig.sizeCellX, self.kConfig.sizeCellY

    @staticmethod
    def _getFwhmPix(exposure):
        sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))
        psf = exposure.getPsf()
        psfAvgPos = psf.getAveragePosition()
        psfSize = psf.computeShape(psfAvgPos).getDeterminantRadius()*sigma2fwhm
        return psfSize
