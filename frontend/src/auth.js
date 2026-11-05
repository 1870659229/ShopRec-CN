import { ref } from 'vue'

// 登录态单一事实源：localStorage 持久化 + ref 响应式。
// 修复 App.vue 挂载时一次性读取 localStorage 导致的
// "登录/退出后顶栏 uid 不更新"（旧 uid 会残留到下次整页刷新）。
const userId = ref(localStorage.getItem('shoprec_user_id') || '')
// 用户名与 user_id 同源持久化：登录/注册响应直接带回，旧 session
// （localStorage 无 username）由 App.vue 调 /api/user/info 补查。
const username = ref(localStorage.getItem('shoprec_username') || '')

export function setUserId(id) {
  localStorage.setItem('shoprec_user_id', String(id))
  userId.value = String(id)
}

export function clearUserId() {
  localStorage.removeItem('shoprec_user_id')
  userId.value = ''
}

export function setUsername(name) {
  localStorage.setItem('shoprec_username', String(name))
  username.value = String(name)
}

export function clearUsername() {
  localStorage.removeItem('shoprec_username')
  username.value = ''
}

export { userId, username }
