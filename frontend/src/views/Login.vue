<template>
  <div class="login-wrap">
    <div class="login-card">
      <div class="login-card__head">
        <span class="login-card__dot" aria-hidden="true"></span>
        <div>
          <h1 class="login-card__title">品智推</h1>
          <p class="login-card__sub">新品与长尾品类冷启动推荐</p>
        </div>
      </div>

      <el-form @submit.prevent>
        <el-form-item label="用户名">
          <!-- 回车 = 登录，用户名/密码两栏行为一致。
               原实现把 Enter 单独绑到「注册」且只绑在密码框：已注册用户填完密码
               按回车会收到 409「用户名已存在」，演示第一步即卡住；且用户名框按
               回车完全无反应。注册改为显式点按钮——这也是登录表单的通用直觉。 -->
          <el-input
            v-model="username"
            placeholder="请输入用户名"
            autofocus
            @keyup.enter="doLogin"
          />
        </el-form-item>
        <el-form-item label="密码">
          <el-input
            v-model="password"
            type="password"
            show-password
            placeholder="请输入密码（至少 6 位）"
            @keyup.enter="doLogin"
          />
        </el-form-item>
      </el-form>

      <div class="login-card__actions">
        <el-button type="primary" class="btn-primary" :loading="loading" @click="doRegister">
          注册
        </el-button>
        <el-button class="btn-plain" :loading="loading" @click="doLogin">登录</el-button>
      </div>

      <p v-if="msg" class="login-card__msg" :class="{ 'is-error': isErr }">{{ msg }}</p>
      <p class="login-card__hint">演示环境 · 注册后即可在推荐流体验冷启动推荐</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { setUserId, setUsername } from '../auth'
import { login, register } from '../api'

const username = ref('')
const password = ref('')
const msg = ref('')
const isErr = ref(false)
const loading = ref(false)
const router = useRouter()

function setMsg(text, error = false) {
  msg.value = text
  isErr.value = error
}

async function doRegister() {
  if (loading.value) return
  loading.value = true
  try {
    // 注册响应即带 user_id/username：直接建立登录态进推荐流，
    // 与"注册后即可体验冷启动"的提示一致（淘宝式注册即登录）
    const { data } = await register({ username: username.value, password: password.value })
    setUserId(data.user_id)
    setUsername(data.username)
    router.push('/feed')
  } catch (e) {
    setMsg(e.response?.data?.detail || '注册失败', true)
  } finally {
    loading.value = false
  }
}

async function doLogin() {
  if (loading.value) return
  loading.value = true
  try {
    const { data } = await login({ username: username.value, password: password.value })
    setUserId(data.user_id)
    setUsername(data.username)
    router.push('/feed')
  } catch (e) {
    setMsg(e.response?.data?.detail || '登录失败', true)
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-wrap {
  min-height: calc(100vh - 56px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
}

.login-card {
  width: 100%;
  max-width: 400px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow-card);
  padding: 28px;
}

.login-card__head {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 24px;
}
.login-card__dot {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--brand);
  box-shadow: 0 0 0 4px rgba(18, 160, 111, 0.16);
  flex: none;
}
.login-card__title {
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  color: var(--ink);
  letter-spacing: 1px;
}
.login-card__sub {
  margin: 2px 0 0;
  font-size: var(--fs-caption);
  color: var(--ink-soft);
}

.login-card__actions {
  display: flex;
  gap: 10px;
  margin-top: 4px;
}
.login-card__actions .btn-primary {
  flex: 1;
  --el-button-bg-color: var(--brand);
  --el-button-border-color: var(--brand);
}
.login-card__actions .btn-plain {
  flex: 1;
  background: var(--surface);
  border: 1px solid var(--line-strong);
  color: var(--ink);
}
.login-card__actions .btn-plain:hover {
  background: var(--bg);
  border-color: var(--ink-faint);
}

.login-card__msg {
  margin: 14px 0 0;
  font-size: 13px;
  color: var(--brand-ink);
}
.login-card__msg.is-error {
  color: var(--danger);
}
.login-card__hint {
  margin: 8px 0 0;
  font-size: var(--fs-caption);
  color: var(--ink-faint);
}
</style>
