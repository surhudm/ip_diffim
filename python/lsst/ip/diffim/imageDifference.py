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

import numpy as np

import lsst.afw.image
import lsst.afw.math
import lsst.geom
import lsst.pex.config
import lsst.pipe.base
from lsst.pipe.base import connectionTypes

__all__ = ["AlardLuptonSubtractConfig", "AlardLuptonSubtractTask"]

_dimensions = ("instrument", "visit", "detector")
_defaultTemplates = {"coaddName": "deep", "fakesType": ""}


class SubtractInputConnections(lsst.pipe.base.PipelineTaskConnections,
                               dimensions=_dimensions,
                               defaultTemplates=_defaultTemplates):
    template = connectionTypes.Input(
        doc="Input warped template to subtract.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_templateExp"
    )
    science = connectionTypes.Input(
        doc="Input science exposure to subtract from.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}calexp"
    )
    sources = connectionTypes.Input(
        doc="Sources measured on the science exposure; "
            "used to select sources for making the matching kernel.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="SourceCatalog",
        name="{fakesType}src"
    )


class SubtractImageOutputConnections(lsst.pipe.base.PipelineTaskConnections,
                                     dimensions=_dimensions,
                                     defaultTemplates=_defaultTemplates):
    difference = connectionTypes.Output(
        doc="Result of subtracting convolved template from science image.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_differenceExp",
    )
    matchedTemplate = connectionTypes.Output(
        doc="Warped and PSF-matched template used to create `subtractedExposure`.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_matchedExp",
    )


class SubtractLikelihoodOutputConnections(lsst.pipe.base.PipelineTaskConnections,
                                          dimensions=_dimensions,
                                          defaultTemplates=_defaultTemplates):
    scoreExposure = connectionTypes.Output(
        doc="Output likelihood score image",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_scoreExp",
    )


class AlardLuptonSubtractConnections(SubtractInputConnections, SubtractImageOutputConnections):
    pass


class AlardLuptonSubtractConfig(lsst.pipe.base.PipelineTaskConfig,
                                pipelineConnections=AlardLuptonSubtractConnections):
    mode = lsst.pex.config.ChoiceField(
        dtype=str,
        default="auto",
        allowed={"auto": "Choose which image to convolve at runtime.",
                 "forceConvolveScience": "Only convolve the science image.",
                 "forceConvolveTemplate": "Only convolve the template image."},
        doc="Choose which image to convolve at runtime, or require that a specific image is convolved."
    )
    makeKernel = lsst.pex.config.ConfigurableField(
        target=lsst.ip.diffim.MakeKernelTask,
        doc="Task to construct a matching kernel for convolution.",
    )
    doDecorrelation = lsst.pex.config.Field(
        dtype=bool,
        default=True,
        doc="Perform diffim decorrelation to undo pixel correlation due to A&L "
        "kernel convolution? If True, also update the diffim PSF."
    )
    decorrelate = lsst.pex.config.ConfigurableField(
        target=lsst.ip.diffim.DecorrelateALKernelTask,
        doc="Task to decorrelate the image difference.",
    )
    requiredTemplateFraction = lsst.pex.config.Field(
        dtype=float,
        default=0.1,
        doc="Do not attempt to run task if template covers less than this fraction of pixels."
        "Setting to 0 will always attempt image subtraction"
    )

    doSelectSources = lsst.pex.config.Field(
        dtype=bool,
        default=False,
        doc="Select stars to use for kernel fitting (compatibility with Gen2 version)"
    )

    def setDefaults(self):
        # defaults are OK for catalog and diacatalog

        self.makeKernel.kernel.name = "AL"
        self.makeKernel.kernel.active.fitForBackground = True
        self.makeKernel.kernel.active.spatialKernelOrder = 1
        self.makeKernel.kernel.active.spatialBgOrder = 2


class AlardLuptonSubtractTask(lsst.pipe.base.PipelineTask):
    """Base class for image subtraction Tasks using the Alard & Lupton (1998)
    algorithm.
    """
    ConfigClass = AlardLuptonSubtractConfig
    _DefaultName = "alardLuptonSubtract"

    def __init__(self, butler=None, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("decorrelate")
        self.makeSubtask("makeKernel")

        self.convolutionControl = lsst.afw.math.ConvolutionControl()
        self.convolutionControl.setDoNormalize(False)

    @lsst.utils.inheritDoc(lsst.pipe.base.PipelineTask)
    def runQuantum(self, butlerQC: lsst.pipe.base.ButlerQuantumContext,
                   inputRefs: lsst.pipe.base.InputQuantizedConnection,
                   outputRefs: lsst.pipe.base.OutputQuantizedConnection):
        inputs = butlerQC.get(inputRefs)
        self.log.info("Processing %s", butlerQC.quantum.dataId)
        checkTemplateIsSufficient(inputs['template'], self.log,
                                  requiredTemplateFraction=self.config.requiredTemplateFraction)

        outputs = self.run(template=inputs['template'],
                           science=inputs['science'],
                           sources=inputs['sources'])
        butlerQC.put(outputs, outputRefs)

    def run(self, template, science, sources):
        """PSF match, subtract, and decorrelate two images.

        Parameters
        ----------
        template : `lsst.afw.image.ExposureF`
            The template image, which has previously been warped and cropped to the science image.
        science : `lsst.afw.image.ExposureF`
            The science exposure.
        sources : `lsst.afw.table.SourceCatalog`
            Identified sources on the science exposure. This catalog is used to
            select sources in order to perform the AL PSF matching on stamp images
            around them.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            ``difference`` : `lsst.afw.image.ExposureF`
                The image difference.
            ``matchedTemplate`` : `lsst.afw.image.ExposureF`
                The warped and PSF-matched template exposure.
        """
        self._validateSize(template, science)
        self._validateWcs(template, science)
        if self.config.doSelectSources:
            # Compatibility option to maintain old functionality
            # This should be removed in the future!
            sources = None
        kernelSources = self.makeKernel.selectKernelSources(template, science,
                                                            candidateList=sources,
                                                            preconvolved=False)
        sciencePsfSize = _getFwhmPix(science)
        templatePsfSize = _getFwhmPix(template)
        self.log.info("Science PSF size: %f", sciencePsfSize)
        self.log.info("Template PSF size: %f", templatePsfSize)
        if self.config.mode == "auto":
            if sciencePsfSize < templatePsfSize:
                self.log.info("Template PSF size is the greater: convolving Science image.")
                subtractResults = self.runConvolveScience(template, science, kernelSources)
            else:
                self.log.info("Science PSF size is the greater: convolving Template image.")
                subtractResults = self.runConvolveTemplate(template, science, kernelSources)
        elif self.config.mode == "forceConvolveTemplate":
            self.log.info("`forceConvolveTemplate` is set: convolving Template image.")
            subtractResults = self.runConvolveTemplate(template, science, kernelSources)
        elif self.config.mode == "forceConvolveScience":
            self.log.info("`forceConvolveScience` is set: convolving Science image.")
            subtractResults = self.runConvolveScience(template, science, kernelSources)
        else:
            raise RuntimeError("Cannot handle AlardLuptonSubtract mode: %s", self.config.mode)
        return subtractResults

    def runConvolveTemplate(self, template, science, sources):
        """Convolve the template image with a PSF-matching kernel and subtract from the science image.

        Parameters
        ----------
        template : `lsst.afw.image.ExposureF`
            The template image, which has previously been warped and cropped to the science image.
        science : `lsst.afw.image.ExposureF`
            The science exposure.
        sources : `lsst.afw.table.SourceCatalog`
            Identified sources on the science exposure. This catalog is used to
            select sources in order to perform the AL PSF matching on stamp images
            around them.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            ``difference`` : `lsst.afw.image.ExposureF`
                The image difference.
            ``matchedTemplate`` : `lsst.afw.image.ExposureF`
                The warped and PSF-matched template exposure.
        """
        difference = science.clone()
        kernelRes = self.makeKernel.run(template, science, sources, preconvolved=False)

        matchedTemplate = self._convolveExposure(template, kernelRes.psfMatchingKernel,
                                                 wcs=science.getWcs(),
                                                 psf=science.getPsf(),
                                                 filterLabel=template.getFilterLabel(),
                                                 photoCalib=science.getPhotoCalib())
        difference = _subtractImages(difference, science.maskedImage, matchedTemplate.maskedImage,
                                     backgroundModel=kernelRes.backgroundModel)
        self.finalize(matchedTemplate, science, difference, kernelRes.psfMatchingKernel,
                      templateMatched=True)

        return lsst.pipe.base.Struct(difference=difference,
                                     matchedTemplate=matchedTemplate,
                                     )

    def runConvolveScience(self, template, science, sources):
        """Convolve the science image with a PSF-matching kernel and subtract the template image.

        Parameters
        ----------
        template : `lsst.afw.image.ExposureF`
            The template image, which has previously been warped and cropped to the science image.
        science : `lsst.afw.image.ExposureF`
            The science exposure.
        sources : `lsst.afw.table.SourceCatalog`
            Identified sources on the science exposure. This catalog is used to
            select sources in order to perform the AL PSF matching on stamp images
            around them.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            ``difference`` : `lsst.afw.image.ExposureF`
                The image difference.
            ``matchedTemplate`` : `lsst.afw.image.ExposureF`
                The warped template exposure. Note that in this case, the template has not been PSF-matched.
        """
        difference = science.clone()
        kernelRes = self.makeKernel.run(template, science, sources, preconvolved=False).psfMatchingKernel

        matchedScience = self._convolveExposure(science, kernelRes.psfMatchingKernel,
                                                wcs=science.getWcs(),
                                                psf=template.getPsf(),
                                                filterLabel=science.getFilterLabel(),
                                                photoCalib=science.getPhotoCalib())
        difference = _subtractImages(difference, matchedScience.maskedImage, template.maskedImage,
                                     psf=template.getPsf(),
                                     backgroundModel=kernelRes.backgroundModel)

        # Place back on native photometric scale
        difference.maskedImage /= kernelRes.psfMatchingKernel.computeImage(
            lsst.afw.image.ImageD(kernelRes.psfMatchingKernel.getDimensions()), False)
        self.finalize(template, matchedScience, difference, kernelRes.psfMatchingKernel,
                      templateMatched=False)

        return lsst.pipe.base.Struct(difference=difference,
                                     matchedTemplate=template,
                                     )

    def finalize(self, template, science, difference, kernel,
                 templateMatched=True,
                 preConvMode=False,
                 preConvKernel=None,
                 spatiallyVarying=False):
        """Decorrelate the difference image to undo the noise correlations caused by convolution.

        Parameters
        ----------
        template : `lsst.afw.image.ExposureF`
            The template image, which has previously been warped and cropped to the science image.
        science : `lsst.afw.image.ExposureF`
            The science exposure.
        difference : `lsst.afw.image.ExposureF`
            The image difference.
        kernel : `lsst.afw.detection.Psf`
            An (optionally spatially-varying) PSF matching kernel
        templateMatched : `bool`, optional
            Was the template PSF-matched to the science image?
        preConvMode : `bool`, optional
            Was the science image preconvolved with its own PSF before PSF matching the template?
        preConvKernel : `lsst.afw.math.Kernel`, optional
            If not `None`, then the `scienceExposure` was pre-convolved with (the reflection of)
            this kernel. Must be normalized to sum to 1.
        spatiallyVarying : `bool`, optional
            Compute the decorrelation kernel spatially varying across the image?
        """
        if self.config.doDecorrelation:
            self.log.info("Decorrelating image difference.")
            self.decorrelate.run(science, template, difference, kernel,
                                 templateMatched=templateMatched,
                                 preConvMode=preConvMode,
                                 preConvKernel=preConvKernel,
                                 spatiallyVarying=spatiallyVarying)
        else:
            self.log.info("NOT decorrelating image difference.")

    def _validateSize(self, template, science):
        """Return True if two image-like objects are the same size.
        """
        templateDims = template.getDimensions()
        scienceDims = science.getDimensions()
        if templateDims != scienceDims:
            raise RuntimeError("Input images different size: template %s vs science %s",
                               templateDims, scienceDims)

    def _validateWcs(self, templateExposure, scienceExposure):
        """Return True if the WCS of the two Exposures have the same origin and extent.
        """
        templateWcs = templateExposure.getWcs()
        scienceWcs = scienceExposure.getWcs()
        templateBBox = templateExposure.getBBox()
        scienceBBox = scienceExposure.getBBox()

        # LLC
        templateOrigin = templateWcs.pixelToSky(lsst.geom.Point2D(templateBBox.getBegin()))
        scienceOrigin = scienceWcs.pixelToSky(lsst.geom.Point2D(scienceBBox.getBegin()))

        # URC
        templateLimit = templateWcs.pixelToSky(lsst.geom.Point2D(templateBBox.getEnd()))
        scienceLimit = scienceWcs.pixelToSky(lsst.geom.Point2D(scienceBBox.getEnd()))

        self.log.info("Template Wcs : %f,%f -> %f,%f",
                      templateOrigin[0], templateOrigin[1],
                      templateLimit[0], templateLimit[1])
        self.log.info("Science Wcs : %f,%f -> %f,%f",
                      scienceOrigin[0], scienceOrigin[1],
                      scienceLimit[0], scienceLimit[1])

        templateBBox = lsst.geom.Box2D(templateOrigin.getPosition(lsst.geom.degrees),
                                       templateLimit.getPosition(lsst.geom.degrees))
        scienceBBox = lsst.geom.Box2D(scienceOrigin.getPosition(lsst.geom.degrees),
                                      scienceLimit.getPosition(lsst.geom.degrees))
        if not (templateBBox.overlaps(scienceBBox)):
            raise RuntimeError("Input images do not overlap at all")

        if ((templateOrigin != scienceOrigin) or (templateLimit != scienceLimit)):
            raise RuntimeError("Template and science exposure WCS are not matched.")

    def _convolveExposure(self, exposure, kernel,
                          wcs=None,
                          psf=None,
                          filterLabel=None,
                          photoCalib=None):
        """Convolve an exposure with the given kernel.

        Parameters
        ----------
        exposure : `lsst.afw.Exposure`
            The exposure to convolve.
        kernel : `lsst.afw.math.LinearCombinationKernel`
            PSF matching kernel computed in the ``makeKernel`` subtask.
        wcs : `lsst.afw.geom.SkyWcs`, optional
            Coordinate system definition (wcs) for the exposure.
        psf : `lsst.afw.detection.Psf`, optional
            Point spread function (PSF) to set for the convolved exposure.
        filterLabel : `lsst.afw.image.FilterLabel`, optional
            The filter label, set in the current instruments' obs package.
        photoCalib : `lsst.afw.image.PhotoCalib`, optional
            Calibration to convert instrumental flux and
            flux error to nanoJansky.

        Returns
        -------
        convolvedExp : `lsst.afw.Exposure`
            The convolved image.
        """
        if wcs is None:
            wcs = exposure.getWcs()
        if psf is None:
            psf = exposure.getPsf()
        if filterLabel is None:
            filterLabel = exposure.getFilterLabel()
        if photoCalib is None:
            photoCalib = exposure.getPhotoCalib()
        convolvedImage = lsst.afw.image.MaskedImageF(exposure.getBBox())
        lsst.afw.math.convolve(convolvedImage, exposure.maskedImage, kernel, self.convolutionControl)
        return _makeExposure(convolvedImage, wcs, psf, filterLabel, photoCalib)


class AlardLuptonPreconvolveSubtractConnections(SubtractInputConnections,
                                                SubtractLikelihoodOutputConnections):
    pass


class AlardLuptonPreconvolveSubtractConfig(lsst.pipe.base.PipelineTaskConfig,
                                           pipelineConnections=AlardLuptonPreconvolveSubtractConnections):
    makeKernel = lsst.pex.config.ConfigurableField(
        target=lsst.ip.diffim.MakeKernelTask,
        doc="Task to construct a matching kernel for convolution.",
    )
    doDecorrelation = lsst.pex.config.Field(
        dtype=bool,
        default=True,
        doc="Perform diffim decorrelation to undo pixel correlation due to A&L "
        "kernel convolution? If True, also update the diffim PSF."
    )
    decorrelate = lsst.pex.config.ConfigurableField(
        target=lsst.ip.diffim.DecorrelateALKernelTask,
        doc="Task to decorrelate the image difference.",
    )
    requiredTemplateFraction = lsst.pex.config.Field(
        dtype=float, default=0.1,
        doc="Do not attempt to run task if template covers less than this fraction of pixels."
        "Setting to 0 will always attempt image subtraction"
    )

    def setDefaults(self):
        self.makeKernel.kernel.name = "AL"
        self.makeKernel.kernel.active.fitForBackground = True
        self.makeKernel.kernel.active.spatialKernelOrder = 1
        self.makeKernel.kernel.active.spatialBgOrder = 2


class AlardLuptonPreconvolveSubtractTask(AlardLuptonSubtractTask):
    """Subtract a template from a science image, convolving the science image
    before computing the kernel, and also convolving the template before
    subtraction.
    """
    ConfigClass = AlardLuptonPreconvolveSubtractConfig
    _DefaultName = "alardLuptonPreconvolveSubtract"

    def run(self, template, science, sources):
        """Summary

        Parameters
        ----------
        template : `lsst.afw.image.ExposureF`
            The template image, which has previously been warped and cropped to the science image.
        science : `lsst.afw.image.ExposureF`
            The science exposure.
        sources : `lsst.afw.table.SourceCatalog`
            Identified sources on the science exposure. This catalog is used to
            select sources in order to perform the AL PSF matching on stamp images
            around them.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            ``scoreExposure`` : `lsst.afw.image.ExposureF`
                The image difference.
            ``matchedTemplate`` : `lsst.afw.image.ExposureF`
                The warped and PSF-matched template exposure.
        """
        sciencePsfSize = _getFwhmPix(science)
        templatePsfSize = _getFwhmPix(template)
        self.log.info("Science PSF size: %f", sciencePsfSize)
        self.log.info("Template PSF size: %f", templatePsfSize)
        # cannot convolve in place, so need a new image anyway
        psfAvgPos = science.psf.getAveragePosition()
        preconvolveKernel = science.psf.getLocalKernel(psfAvgPos)
        scoreExposure = science.clone()
        convolvedScience = self._convolveExposure(science, preconvolveKernel)

        kernelRes = self.makeKernel.run(template, convolvedScience, sources, preconvolved=True)

        self.log.info("Preconvolving science image, and convolving template image")
        matchedTemplate = self._convolveExposure(template, kernelRes.psfMatchingKernel)
        scoreExposure = _subtractImages(scoreExposure, convolvedScience, matchedTemplate)
        self.finalize(matchedTemplate, science, scoreExposure, kernelRes.psfMatchingKernel,
                      templateMatched=True,
                      preConvMode=True,
                      preConvKernel=preconvolveKernel)

        return lsst.pipe.base.Struct(scoreExposure=scoreExposure,
                                     matchedTemplate=matchedTemplate,
                                     )


def _getFwhmPix(exposure):
    sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))
    psf = exposure.getPsf()
    psfAvgPos = psf.getAveragePosition()
    psfSize = psf.computeShape(psfAvgPos).getDeterminantRadius()*sigma2fwhm
    return psfSize


def _makeExposure(maskedImage, wcs, psf, filterLabel, photoCalib):
    newExposure = lsst.afw.image.ExposureF(maskedImage, wcs)
    newExposure.setPsf(psf)
    newExposure.setFilterLabel(filterLabel)
    newExposure.setPhotoCalib(photoCalib)
    return newExposure


def checkTemplateIsSufficient(templateExposure, logger, requiredTemplateFraction=0.):
    """Raise NoWorkFound if template coverage < requiredTemplateFraction

    Parameters
    ----------
    templateExposure : `lsst.afw.image.ExposureF`
        The template exposure to check
    logger : `lsst.log.Log`
        Logger for printing output.
    requiredTemplateFraction : `float`, optional
        Fraction of pixels of the science image required to have coverage
        in the template.

    Raises
    ------
    lsst.pipe.base.NoWorkFound
        Raised if fraction of good pixels, defined as not having NO_DATA
        set, is less then the configured requiredTemplateFraction
    """
    # Count the number of pixels with the NO_DATA mask bit set
    # counting NaN pixels is insufficient because pixels without data are often intepolated over)
    pixNoData = np.count_nonzero(templateExposure.mask.array
                                 & templateExposure.mask.getPlaneBitMask('NO_DATA'))
    pixGood = templateExposure.getBBox().getArea() - pixNoData
    logger.info("template has %d good pixels (%.1f%%)", pixGood,
                100*pixGood/templateExposure.getBBox().getArea())

    if pixGood/templateExposure.getBBox().getArea() < requiredTemplateFraction:
        message = ("Insufficient Template Coverage. (%.1f%% < %.1f%%) Not attempting subtraction. "
                   "To force subtraction, set config requiredTemplateFraction=0." % (
                       100*pixGood/templateExposure.getBBox().getArea(),
                       100*requiredTemplateFraction))
        raise lsst.pipe.base.NoWorkFound(message)


def _subtractImages(differenceExp, science, template, psf=None, backgroundModel=None):
    """Subtract template from science, propagating relevant metadata.

    Parameters
    ----------
    differenceExp : `lsst.afw.Exposure`
        The output exposure that will contain the subtracted image.
    science : `lsst.afw.MaskedImage`
        The input science image.
    template : `lsst.afw.MaskedImage`
        The template to subtract from the science image.
    psf : `lsst.afw.detection.Psf`, optional
        The PSF to set for the difference image.
    backgroundModel : `lsst.afw.MaskedImage`, optional
        Differential background model

    Returns
    -------
    differenceExp : `lsst.afw.Exposure`
        The subtracted image.
    """
    differenceExp.maskedImage = science
    if backgroundModel is not None:
        science -= backgroundModel
    science -= template
    if psf is not None:
        differenceExp.setPsf(psf)
    return differenceExp
