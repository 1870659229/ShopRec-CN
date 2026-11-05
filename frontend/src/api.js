import axios from 'axios'

// 字段名 → 中文标签（Pydantic 校验错误 loc 第二段即字段名）
const FIELD_LABELS = {
  username: '用户名',
  password: '密码',
  user_id: '用户 ID',
  item_id: '商品 ID',
  behavior_type: '行为类型',
  rating: '评分'
}

// 单条 Pydantic 校验错误 → 中文可读短句
function describeError(err) {
  const type = err.type || ''
  const ctx = err.ctx || {}
  const field = (err.loc || []).slice(1).join('.') || ''
  const label = FIELD_LABELS[field] || field || '参数'
  if (type === 'string_too_short') return `${label}至少 ${ctx.min_length} 位`
  if (type === 'string_too_long') return `${label}至多 ${ctx.max_length} 位`
  if (type === 'missing' || type === 'anystr') return `${label}不能为空`
  if (type === 'int_too_big') return `${label}超出上限 ${ctx.max}`
  if (type === 'int_too_small') return `${label}低于下限 ${ctx.min}`
  if (type === 'greater_than') return `${label}需大于 ${ctx.gt}`
  if (type === 'greater_than_equal') return `${label}需不小于 ${ctx.ge}`
  if (type === 'less_than') return `${label}需小于 ${ctx.lt}`
  if (type === 'less_than_equal') return `${label}需不大于 ${ctx.le}`
  return `${label}：${err.msg || '格式不正确'}`
}

const api = axios.create({ baseURL: '/api', timeout: 15000 })

// FastAPI 校验失败（422）时 detail 是 Pydantic 错误数组，各页面
// `e.response?.data?.detail || '…'` 会把整段 JSON 渲染出来。这里就地归一为中文短句。
api.interceptors.response.use(
  (res) => res,
  (error) => {
    const data = error.response?.data
    if (error.response?.status === 422 && Array.isArray(data?.detail)) {
      data.detail = data.detail.map(describeError).join('；')
    }
    return Promise.reject(error)
  }
)

export const register = (d) => api.post('/user/register', d)
export const login = (d) => api.post('/user/login', d)
export const fetchUserInfo = (userId) => api.get('/user/info', { params: { user_id: userId } })
export const fetchUserBehaviors = (params) => api.get('/user/behaviors', { params })
export const fetchRecommend = (userId, k = 10) =>
  api.get('/recommend', { params: { user_id: userId, k } })
export const sendFeedback = (d) => api.post('/feedback', d)
export const checkUpdate = () => api.post('/update/check')
export const fetchEvalResults = () => api.get('/eval/results')
export const fetchCategories = () => api.get('/category/list')
export const searchItems = (params) => api.get('/item/search', { params })
export const suggestItems = (q) => api.get('/item/suggest', { params: { q } })

export default api
