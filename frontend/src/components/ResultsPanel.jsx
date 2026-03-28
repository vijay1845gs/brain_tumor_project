import { Brain, AlertTriangle, CheckCircle, Info, Microscope, FileText,
         Eye, Download, BarChart3, Zap, GitCompare } from 'lucide-react'
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'
import ConfidenceBar from './ConfidenceBar'
import RiskBadge from './RiskBadge'
import { predictionService } from '../services/api'

const TUMOR_META = {
  glioma:      { label: 'Glioma',           color: 'text-red-600',    bg: 'bg-red-50',    border: 'border-red-100'    },
  meningioma:  { label: 'Meningioma',        color: 'text-orange-600', bg: 'bg-orange-50', border: 'border-orange-100' },
  pituitary:   { label: 'Pituitary Adenoma', color: 'text-yellow-600', bg: 'bg-yellow-50', border: 'border-yellow-100' },
}

const TUMOR_BAR_COLORS = {
  glioma:     '#ef4444',
  meningioma: '#f97316',
  pituitary:  '#eab308',
}

function ClassProbBar({ label, prob, maxProb }) {
  const pct = maxProb > 0 ? (prob / maxProb) * 100 : 0
  const color = TUMOR_BAR_COLORS[label] || '#6366f1'
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-slate-500 capitalize w-20 shrink-0">{label}</span>
      <div className="flex-1 h-2.5 bg-slate-100 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
        />
      </div>
      <span className="text-xs font-bold text-slate-600 w-12 text-right shrink-0">
        {(prob * 100).toFixed(1)}%
      </span>
    </div>
  )
}

function UncertaintyGauge({ value, max = 0.3, label }) {
  const pct = Math.min((value / max) * 100, 100)
  const isHigh = value > 0.1
  return (
    <div>
      <div className="flex justify-between items-end mb-1">
        <span className="text-xs font-medium text-slate-500">{label}</span>
        <span className={`text-xs font-mono font-bold ${isHigh ? 'text-amber-600' : 'text-emerald-600'}`}>
          ±{value.toFixed(4)}
        </span>
      </div>
      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8 }}
          className={`h-full rounded-full ${isHigh ? 'bg-amber-400' : 'bg-emerald-400'}`}
        />
      </div>
    </div>
  )
}

const CAM_TABS = [
  { id: 'gradcam',  icon: Zap,        label: 'Grad-CAM++', field: 'heatmap_gradcam',  desc: 'Gradient-weighted activation map highlighting decision regions.' },
  { id: 'eigencam', icon: GitCompare, label: 'EigenCAM',   field: 'heatmap_eigencam', desc: 'PCA on feature maps — robust when gradients are unstable.' },
]

export default function ResultsPanel({ result }) {
  const [downloading, setDownloading] = useState(false)
  const [heatmapMode, setHeatmapMode] = useState('gradcam')
  const [opacity, setOpacity] = useState(0.7)

  const {
    tumor_detected, tumor_type, confidence, uncertainty, reliability,
    risk_level, clinical_note, recommendation, all_class_probs,
    heatmap_gradcam, heatmap_scorecam, heatmap_eigencam, heatmap_image,
  } = result

  const tumorMeta = tumor_type ? TUMOR_META[tumor_type] : null
  const maxProb = all_class_probs ? Math.max(...Object.values(all_class_probs)) : 1

  // available tabs — only show if backend returned that heatmap
  const availableTabs = CAM_TABS.filter(t => !!result[t.field])
  const activeTab = availableTabs.find(t => t.id === heatmapMode) || availableTabs[0]
  const currentHeatmap = activeTab ? result[activeTab.field] : heatmap_image
  const primaryHeatmap = heatmap_gradcam || heatmap_scorecam || heatmap_eigencam || heatmap_image

  const handleDownloadReport = async () => {
    setDownloading(true)
    try {
      const blob = await predictionService.getReport(result)
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `NeuroScan_Report_${tumor_type || 'NoTumor'}.pdf`)
      document.body.appendChild(link)
      link.click()
      link.remove()
      toast.success('Report downloaded successfully')
    } catch {
      toast.error('Failed to generate report')
    } finally {
      setDownloading(false)
    }
  }

  return (
    <div className="space-y-5 animate-slide-up">

      {/* Download */}
      <div className="flex justify-end">
        <button
          onClick={handleDownloadReport}
          disabled={downloading}
          className="btn-secondary py-2 text-xs transition-all hover:ring-2 hover:ring-neural-100"
        >
          {downloading ? 'Generating...' : <><Download className="w-3.5 h-3.5" /> Download Clinical Report</>}
        </button>
      </div>

      {/* Detection banner */}
      <div className={`flex items-center gap-3 p-4 rounded-2xl border ${
        tumor_detected ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'
      }`}>
        {tumor_detected
          ? <AlertTriangle className="w-6 h-6 text-red-600 shrink-0" />
          : <CheckCircle className="w-6 h-6 text-green-600 shrink-0" />
        }
        <div>
          <p className={`font-semibold text-sm ${tumor_detected ? 'text-red-800' : 'text-green-800'}`}>
            {tumor_detected ? '⚠️ Tumor Detected' : '✅ No Tumor Detected'}
          </p>
          <p className={`text-xs mt-0.5 ${tumor_detected ? 'text-red-600' : 'text-green-600'}`}>
            {reliability}
          </p>
        </div>
      </div>

      {/* Tumor type */}
      {tumor_detected && tumorMeta && (
        <div className={`p-4 rounded-2xl border ${tumorMeta.bg} ${tumorMeta.border}`}>
          <div className="flex items-center gap-2 mb-1">
            <Microscope className={`w-4 h-4 ${tumorMeta.color}`} />
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Tumor Type</span>
          </div>
          <p className={`font-display text-2xl ${tumorMeta.color}`}>{tumorMeta.label}</p>
        </div>
      )}

      {/* Confidence + Uncertainty */}
      <div className="card space-y-4">
        <div>
          <div className="flex justify-between items-end mb-1.5">
            <span className="text-xs font-semibold text-slate-500 uppercase">Detection Confidence</span>
            <span className="text-lg font-display text-neural-600">{(confidence * 100).toFixed(1)}%</span>
          </div>
          <ConfidenceBar value={confidence} />
        </div>

        <div className="grid grid-cols-2 gap-3 pt-3 border-t border-slate-50">
          {/* MC Dropout Uncertainty */}
          <UncertaintyGauge value={uncertainty} max={0.3} label="MC Dropout Uncertainty" />

          {/* Shannon Entropy */}
          {result.entropy !== undefined && (() => {
            const entropyPct = Math.round((result.entropy || 0) * 100)
            return (
              <div>
                <div className="flex justify-between items-end mb-1">
                  <span className="text-xs font-medium text-slate-500">Shannon Entropy</span>
                  <span className={`text-xs font-mono font-bold ${
                    entropyPct > 50 ? 'text-amber-600' : 'text-emerald-600'
                  }`}>{entropyPct}%</span>
                </div>
                <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${entropyPct}%` }}
                    transition={{ duration: 0.8 }}
                    className={`h-full rounded-full ${
                      entropyPct > 50 ? 'bg-amber-400' : 'bg-emerald-400'
                    }`}
                  />
                </div>
              </div>
            )
          })()}
        </div>

        {/* TTA Agreement */}
        {result.tta_agreement !== undefined && (() => {
          const ttaPct = Math.round((result.tta_agreement || 1) * 100)
          return (
            <div className="flex items-center justify-between pt-2 border-t border-slate-50">
              <span className="text-xs font-medium text-slate-500">TTA View Agreement</span>
              <span className={`text-xs font-mono font-bold ${
                ttaPct < 70 ? 'text-amber-600' : 'text-emerald-600'
              }`}>{ttaPct}%</span>
            </div>
          )
        })()}
      </div>

      {/* Class Probability Bars */}
      {all_class_probs && Object.keys(all_class_probs).length > 0 && (
        <div className="card space-y-3">
          <div className="flex items-center gap-2 mb-1">
            <BarChart3 className="w-4 h-4 text-neural-600" />
            <h3 className="font-semibold text-slate-700 text-sm">Class Probability Distribution</h3>
          </div>
          {Object.entries(all_class_probs)
            .sort(([, a], [, b]) => b - a)
            .map(([cls, prob]) => (
              <ClassProbBar key={cls} label={cls} prob={prob} maxProb={maxProb} />
            ))}
        </div>
      )}

      {/* Risk */}
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-slate-600">Risk Assessment</span>
        <RiskBadge level={risk_level} />
      </div>

      {/* Explainability — 3 CAM tabs */}
      {primaryHeatmap && (
        <div className="card space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Eye className="w-4 h-4 text-neural-600" />
              <h3 className="font-semibold text-slate-700 text-sm">Explainability Maps</h3>
            </div>

            {/* Tab switcher — only show available tabs */}
            {availableTabs.length > 1 && (
              <div className="flex items-center gap-1 bg-slate-100 rounded-xl p-1">
                {availableTabs.map(({ id, icon: Icon, label }) => (
                  <button
                    key={id}
                    onClick={() => setHeatmapMode(id)}
                    className={`flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      (activeTab?.id === id)
                        ? 'bg-white text-neural-700 shadow-sm'
                        : 'text-slate-500 hover:text-slate-700'
                    }`}
                  >
                    <Icon className="w-3 h-3" />
                    {label}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Intensity slider */}
          <div className="flex items-center gap-3">
            <label className="text-[10px] uppercase font-bold text-slate-400 shrink-0">Intensity</label>
            <input
              type="range" min="0" max="1" step="0.01"
              value={opacity} onChange={e => setOpacity(parseFloat(e.target.value))}
              className="flex-1 accent-neural-500"
            />
          </div>

          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab?.id}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="relative rounded-xl overflow-hidden bg-slate-100 border border-slate-100 aspect-square"
            >
              <img
                src={currentHeatmap}
                alt={`${activeTab?.label} heatmap`}
                style={{ filter: `saturate(${opacity * 2}) brightness(${0.8 + opacity * 0.2})` }}
                className="w-full h-full object-cover transition-all duration-300"
              />
            </motion.div>
          </AnimatePresence>

          <p className="text-xs text-slate-400">{activeTab?.desc}</p>
        </div>
      )}

      {/* Clinical note */}
      <div className="card border-l-4 border-l-neural-400">
        <div className="flex items-center gap-2 mb-2">
          <Brain className="w-4 h-4 text-neural-600" />
          <h3 className="font-semibold text-slate-700 text-sm">Clinical Note</h3>
        </div>
        <p className="text-sm text-slate-600 leading-relaxed whitespace-pre-line">{clinical_note}</p>
      </div>

      {/* Recommendation */}
      <div className="card border-l-4 border-l-emerald-400">
        <div className="flex items-center gap-2 mb-2">
          <FileText className="w-4 h-4 text-emerald-600" />
          <h3 className="font-semibold text-slate-700 text-sm">Clinical Recommendations</h3>
        </div>
        <p className="text-sm text-slate-600 leading-relaxed whitespace-pre-line">{recommendation}</p>
      </div>

      {/* Disclaimer */}
      <div className="flex items-start gap-2 p-3 bg-slate-50 rounded-xl">
        <Info className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
        <p className="text-xs text-slate-400 leading-relaxed">
          This AI system (EfficientNet-B4 + MC Dropout + Grad-CAM++ + Score-CAM + EigenCAM) is intended
          as a decision-support tool only. All results must be reviewed by a qualified medical professional
          before any clinical action is taken.
        </p>
      </div>
    </div>
  )
}
