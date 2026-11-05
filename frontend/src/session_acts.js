import { ref } from 'vue'

// 本次会话行为流水（客户端演示可见性）：跨商品详情页累积，
// 让"行为 → 画像 → 推荐"闭环可被评委直观看到。
// 权威记录在后端 behavior 表；此处仅为演示可见性，刷新页面即清空。
const acts = ref([])

export function pushAct(label) {
  const now = new Date()
  const t = [now.getHours(), now.getMinutes(), now.getSeconds()]
    .map((x) => String(x).padStart(2, '0'))
    .join(':')
  acts.value.unshift({ t, label })
}

export { acts }
