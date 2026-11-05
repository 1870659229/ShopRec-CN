<template>
  <section>
    <div class="dash-head">
      <h2 class="dash-head__title">后台看板</h2>
      <p class="dash-head__sub">离线评估真实数据 · 非编造指标</p>
    </div>

    <!-- KPI 大数字卡：数值与 delta 全部由 /api/eval/results 计算，不硬编码 -->
    <div v-if="kpis.length" class="dash-kpis">
      <div v-for="k in kpis" :key="k.key" class="dash-kpi">
        <span class="dash-kpi__label">{{ k.label }}</span>
        <span class="dash-kpi__value">{{ k.value }}</span>
        <span class="dash-kpi__delta" :class="k.tone">{{ k.delta }}</span>
      </div>
    </div>

    <el-alert
      v-if="err"
      type="warning"
      :title="err"
      :closable="false"
      show-icon
      class="dash-err"
    />

    <div v-show="!err" class="dash-grid">
      <div class="dash-card">
        <div ref="chart1" class="dash-card__chart"></div>
      </div>
      <div class="dash-card">
        <div ref="chart2" class="dash-card__chart"></div>
      </div>
    </div>

    <p v-if="sourceNote" class="dash-note num">{{ sourceNote }}</p>

    <div class="dash-attach">
      <h3 class="dash-attach__title">离线评估附图（α 扫描导出）</h3>
      <div class="dash-attach__row">
        <figure class="dash-attach__fig">
          <el-image
            class="dash-attach__img"
            :src="'/eval/alpha_scan_pareto.png'"
            :preview-src-list="['/eval/alpha_scan_pareto.png']"
            fit="contain"
            alt="α 扫描 Pareto 前沿：Recall 与 Diversity 的权衡曲线"
            lazy
          />
          <figcaption>α 扫描 · Pareto 前沿（点按放大）</figcaption>
        </figure>
        <figure class="dash-attach__fig">
          <el-image
            class="dash-attach__img"
            :src="'/eval/alpha_scan_tradeoff.png'"
            :preview-src-list="['/eval/alpha_scan_tradeoff.png']"
            fit="contain"
            alt="α 扫描准确率–多样性权衡：Recall@10 与 Diversity@10 随 α 变化"
            lazy
          />
          <figcaption>α 扫描 · 准确率–多样性权衡（点按放大）</figcaption>
        </figure>
      </div>
      <p class="dash-attach__note">
        两张图为离线评估脚本导出的 PNG（300 dpi），数据与上方交互图同源
        （<code>data/splits/jdsearch_eval/alpha_scan.json</code>），非编造指标。
      </p>
    </div>

    <div class="dash-retrain">
      <el-button type="primary" class="btn-primary" @click="triggerUpdate">
        检查是否触发重训
      </el-button>
      <p v-if="updateMsg" class="dash-retrain__msg">{{ updateMsg }}</p>
    </div>
  </section>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'
import * as echarts from 'echarts'
import { checkUpdate, fetchEvalResults } from '../api'

const chart1 = ref(null)
const chart2 = ref(null)
const err = ref('')
const updateMsg = ref('')
const sourceNote = ref('')
// 收集图表实例：窗口缩放时统一 resize（缺此监听时画布不随容器重绘）
const charts = []

function track(instance) {
  charts.push(instance)
  return instance
}

const onResize = () => charts.forEach((c) => c.resize())

// KPI 卡数据：从 metrics 动态推导（ours 键名不写死，接口换版本自动跟随）
const kpis = ref([])

function fmtMetric(v) {
  return Number(v).toFixed(4)
}

function oursKeyOf(metrics) {
  return Object.keys(metrics).find((k) => k.startsWith('ours')) || ''
}

function buildKpis(metrics) {
  const oursKey = oursKeyOf(metrics)
  const ours = oursKey ? metrics[oursKey] : null
  const cf = metrics.cf
  if (!ours || !cf) return []
  const defs = [
    { key: 'recall@10', label: 'Recall@10' },
    { key: 'ndcg@10', label: 'NDCG@10' },
    { key: 'diversity@10', label: 'Diversity@10' },
    { key: 'cold_category_hit_rate', label: '冷启动品类命中率' },
  ]
  return defs
    .map((d) => {
      const v = ours[d.key]
      const base = cf[d.key]
      if (v === undefined || base === undefined) return null
      // ColdHit：CF 基线恒 0，百分比无意义 → 标注"唯一非零方法"
      if (base === 0) {
        return {
          key: d.key,
          label: d.label,
          value: fmtMetric(v),
          delta: v > 0 ? '唯一非零方法' : '与基线持平',
          tone: v > 0 ? 'is-up' : 'is-flat',
        }
      }
      const pct = ((v - base) / base) * 100
      const up = pct >= 0
      return {
        key: d.key,
        label: d.label,
        value: fmtMetric(v),
        delta: `${up ? '▲' : '▼'} ${Math.abs(pct).toFixed(1)}% vs CF`,
        tone: up ? 'is-up' : 'is-down',
      }
    })
    .filter(Boolean)
}

// 看板图表配色：从全局 token 取值，保持与页面一致（玉绿主色 + 墨色坐标）。
// ECharts 不吃 CSS 变量，只能硬编码——与 theme.css 对应 token 同步改，
// 改主题时这两处必须一起动（INK_FAINT 同步于 --ink-faint #6b7480）
const BRAND = '#12a06f'
const INK = '#5b6470'
const INK_FAINT = '#6b7480'

function protocolNote(protocol) {
  if (!protocol) return ''
  const { n_users: n = '?', holdout = '?', seed = '?' } = protocol
  return `数据来源：JDsearch ${n} 用户 leave-last-${holdout}-out 离线评估，seed=${seed}`
}

function renderCompare(metrics) {
  // 分组柱状：x 轴为指标，系列为方法（全部为离线评估真实数据）
  const inds = ['recall@10', 'ndcg@10', 'diversity@10', 'cold_category_hit_rate']
  const indNames = ['Recall@10', 'NDCG@10', 'Diversity@10', '冷启动品类命中率']
  const oursKey = oursKeyOf(metrics) || 'ours_v4'
  const methods = ['pop', 'cf', oursKey]
  const nameMap = {
    pop: '热门基线 Pop',
    cf: '协同过滤 CF',
    // 'ours' 为接口归一化键（eval_v5.json 的 ours@0.05 冻结档）
    [oursKey]: oursKey === 'ours' ? '本方案（α=0.05）' : `本方案 ${oursKey.replace('ours_v', 'v')}`,
  }
  const methodColors = { pop: '#c2cad4', cf: '#7fb6d9', [oursKey]: BRAND }
  const instance = track(echarts.init(chart1.value))
  instance.setOption({
    title: {
      text: '四项指标对比：多样性拉满，覆盖率有代价',
      subtext: compareSubtext(metrics, oursKey),
      left: 'center',
      textStyle: { fontSize: 14, color: '#171a21', fontWeight: 600 },
      subtextStyle: { fontSize: 11, color: INK_FAINT },
    },
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0, data: methods.map((m) => nameMap[m]), textStyle: { color: INK } },
    grid: { top: 76, bottom: 70 },
    xAxis: {
      type: 'category',
      data: indNames,
      axisLabel: { color: INK },
      axisLine: { lineStyle: { color: '#d5d9df' } }
    },
    yAxis: {
      type: 'value',
      name: '指标值',
      axisLabel: { color: INK, formatter: (v) => v.toFixed(3) },
      splitLine: { lineStyle: { color: '#eef0f3' } }
    },
    series: methods.map((m) => ({
      name: nameMap[m],
      type: 'bar',
      barWidth: 22,
      itemStyle: { color: methodColors[m], borderRadius: [4, 4, 0, 0] },
      data: inds.map((i) => metrics[m]?.[i] ?? 0),
      // 柱顶数值标签：评委不必 hover 即可读精确值
      label: {
        show: true,
        position: 'top',
        fontSize: 10,
        color: INK,
        formatter: (p) => Number(p.value).toFixed(4),
      },
    }))
  })
}

// 结论式副标题：由数据动态计算，指标换版不漂移
function compareSubtext(metrics, oursKey) {
  const ours = metrics[oursKey]
  const cf = metrics.cf
  if (!ours || !cf) return ''
  const rPct = ((ours['recall@10'] - cf['recall@10']) / cf['recall@10']) * 100
  return `R@10 ${fmtMetric(ours['recall@10'])} vs CF ${fmtMetric(cf['recall@10'])}（${rPct.toFixed(1)}%）· Div@10 ${fmtMetric(ours['diversity@10'])} · ColdHit ${fmtMetric(ours['cold_category_hit_rate'])}（唯一非零）`
}

function renderAlphaScan(scan, baselines) {
  const alphas = scan.map((r) => r.alpha)
  const cfRecall = baselines?.cf?.['recall@10']
  const cfDiv = baselines?.cf?.['diversity@10']
  const instance = track(echarts.init(chart2.value))
  instance.setOption({
    title: {
      text: 'α 扫描：准确率–多样性权衡',
      subtext: alphas.length
        ? `α 范围 ${alphas[0]} ~ ${alphas[alphas.length - 1]} · 灰色虚线为 CF 基线`
        : '',
      left: 'center',
      textStyle: { fontSize: 14, color: '#171a21', fontWeight: 600 },
      subtextStyle: { fontSize: 11, color: INK_FAINT },
    },
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0, data: ['Recall@10', 'Diversity@10'], textStyle: { color: INK } },
    grid: { top: 76, bottom: 70 },
    xAxis: {
      type: 'category',
      name: 'α',
      data: alphas,
      axisLabel: { color: INK, formatter: (v) => Number(v).toFixed(2) },
      axisLine: { lineStyle: { color: '#d5d9df' } }
    },
    yAxis: [
      {
        type: 'value',
        name: 'Recall@10',
        axisLabel: { color: INK, formatter: (v) => v.toFixed(3) },
        splitLine: { lineStyle: { color: '#eef0f3' } }
      },
      {
        type: 'value',
        name: 'Diversity@10',
        max: 1.05,
        axisLabel: { color: INK },
        splitLine: { show: false }
      }
    ],
    series: [
      {
        name: 'Recall@10',
        type: 'line',
        yAxisIndex: 0,
        smooth: true,
        symbolSize: 7,
        lineStyle: { width: 3, color: BRAND },
        itemStyle: { color: BRAND },
        data: scan.map((r) => r['recall@10']),
        ...(cfRecall !== undefined
          ? {
              markLine: {
                symbol: 'none',
                data: [{ yAxis: cfRecall, name: 'CF 基线' }],
                lineStyle: { type: 'dashed', color: INK_FAINT },
                label: { formatter: 'CF 基线 ' + cfRecall, color: INK_FAINT }
              }
            }
          : {})
      },
      {
        name: 'Diversity@10',
        type: 'line',
        yAxisIndex: 1,
        smooth: true,
        symbolSize: 7,
        lineStyle: { width: 3, color: '#7fb6d9' },
        itemStyle: { color: '#7fb6d9' },
        data: scan.map((r) => r['diversity@10']),
        ...(cfDiv !== undefined
          ? {
              markLine: {
                symbol: 'none',
                data: [{ yAxis: cfDiv, name: 'CF 基线' }],
                lineStyle: { type: 'dashed', color: INK_FAINT },
                label: { formatter: 'CF 基线 ' + cfDiv, color: INK_FAINT }
              }
            }
          : {})
      }
    ]
  })
}

onMounted(async () => {
  try {
    const { data } = await fetchEvalResults()
    const { eval_v5: ev, alpha_scan: as } = data
    sourceNote.value = protocolNote(ev?.protocol)
    kpis.value = buildKpis(ev?.metrics || {})
    renderCompare(ev?.metrics || {})
    renderAlphaScan(as?.alpha_scan || [], as?.baselines)
  } catch (e) {
    // 接口 404 / 网络异常：展示兜底文案，不白屏
    err.value =
      e.response?.status === 404
        ? '离线评估结果文件尚未生成（data/splits/jdsearch_eval/），请先运行离线评估流水线'
        : `看板数据加载失败：${e.response?.data?.detail || e.message || '未知错误'}`
  }
  window.addEventListener('resize', onResize)
})
onBeforeUnmount(() => window.removeEventListener('resize', onResize))

async function triggerUpdate() {
  try {
    const { data } = await checkUpdate()
    updateMsg.value = data.retrain_triggered
      ? `已触发重训，当前本地行为 ${data.count} 条`
      : `未达阈值，当前本地行为 ${data.count} 条`
  } catch (e) {
    updateMsg.value = e.response?.data?.detail || '重训检查失败'
  }
}
</script>

<style scoped>
.dash-head {
  margin-bottom: 20px;
}
.dash-head__title {
  margin: 0;
  font-size: var(--fs-hero);
  font-weight: 700;
  color: var(--ink);
}
.dash-head__sub {
  margin: 6px 0 0;
  font-size: 13px;
  color: var(--ink-soft);
}
.dash-err {
  margin-bottom: 16px;
}

/* KPI 大数字卡：先给结论，再给图表（生意参谋式信息层级） */
.dash-kpis {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 16px;
}
@media (max-width: 720px) {
  .dash-kpis {
    grid-template-columns: repeat(2, 1fr);
  }
}
.dash-kpi {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow-card);
  padding: 14px 16px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.dash-kpi__label {
  font-size: var(--fs-caption);
  color: var(--ink-soft);
}
.dash-kpi__value {
  font-size: 24px;
  font-weight: 700;
  color: var(--ink);
  line-height: 1.2;
}
.dash-kpi__delta {
  font-size: var(--fs-caption);
}
/* 涨落用语义色（--ok/--warn），不用红绿，避开股市配色歧义 */
.dash-kpi__delta.is-up {
  color: var(--brand-ink);
}
.dash-kpi__delta.is-down {
  color: var(--warn);
}
.dash-kpi__delta.is-flat {
  color: var(--ink-faint);
}

.dash-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
@media (max-width: 720px) {
  .dash-grid {
    grid-template-columns: 1fr;
  }
}

.dash-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow-card);
  padding: 16px;
}
.dash-card__chart {
  height: 340px;
}

.dash-note {
  margin: 16px 0 0;
  font-size: var(--fs-caption);
  color: var(--ink-faint);
}

.dash-retrain {
  margin-top: 20px;
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}
.dash-retrain .btn-primary {
  --el-button-bg-color: var(--brand);
  --el-button-border-color: var(--brand);
}
.dash-retrain__msg {
  margin: 0;
  font-size: 13px;
  color: var(--ink-soft);
}

/* 离线评估附图区 */
.dash-attach {
  margin-top: 24px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow-card);
  padding: 18px 20px 20px;
}
.dash-attach__title {
  margin: 0 0 14px;
  font-size: 15px;
  font-weight: 600;
  color: var(--ink);
}
.dash-attach__row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
@media (max-width: 720px) {
  .dash-attach__row {
    grid-template-columns: 1fr;
  }
}
.dash-attach__fig {
  margin: 0;
}
.dash-attach__img {
  width: 100%;
  height: 300px;
  background: #fbfcfd;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  overflow: hidden;
}
.dash-attach__fig figcaption {
  margin-top: 8px;
  font-size: var(--fs-caption);
  color: var(--ink-soft);
  text-align: center;
}
.dash-attach__note {
  margin: 14px 0 0;
  font-size: var(--fs-caption);
  color: var(--ink-faint);
}
.dash-attach__note code {
  background: var(--bg);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 1px 5px;
  font-size: 11px;
  color: var(--ink-soft);
}
</style>
