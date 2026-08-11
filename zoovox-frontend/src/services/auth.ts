import {
  authApi,
  captureFrame,
  token,
  type AuthResponse,
  type User,
} from "./api";

// ─────────────────────────────────────────────────────────────
// LOGIN
// ─────────────────────────────────────────────────────────────
export const login = async (
  email: string,
  password: string
): Promise<AuthResponse> => {

  return await authApi.login(
    email,
    password
  );
};

// ─────────────────────────────────────────────────────────────
// REGISTER
// ─────────────────────────────────────────────────────────────
export const register = async (
  name: string,
  email: string,
  password: string
): Promise<AuthResponse> => {

  return await authApi.register(
    name,
    email,
    password
  );
};

// ─────────────────────────────────────────────────────────────
// FACE LOGIN
// ─────────────────────────────────────────────────────────────
export const faceLogin = async (
  frameB64: string
): Promise<AuthResponse> => {

  return await authApi.faceLogin(
    frameB64
  );
};

// ─────────────────────────────────────────────────────────────
// FACE ENROLLMENT
// ─────────────────────────────────────────────────────────────
export const enrollFace = async (
  frameB64: string
): Promise<{
  success: boolean;
  message: string;
}> => {

  return await authApi.enrollFace(
    frameB64
  );
};

// ─────────────────────────────────────────────────────────────
// CURRENT USER
// ─────────────────────────────────────────────────────────────
export const getCurrentUser =
  async (): Promise<User> => {

    return await authApi.getMe();
};

// ─────────────────────────────────────────────────────────────
// LOGOUT
// ─────────────────────────────────────────────────────────────
export const logout = () => {

  authApi.logout();
};

// ─────────────────────────────────────────────────────────────
// TOKEN HELPERS
// ─────────────────────────────────────────────────────────────
export const getAccessToken =
  () => token.get();

export const getRefreshToken =
  () => token.getRefresh();

export const getStoredUser =
  () => token.getUser();

// ─────────────────────────────────────────────────────────────
// CAMERA FRAME HELPER
// ─────────────────────────────────────────────────────────────
export const captureFaceFrame =
  (
    video: HTMLVideoElement
  ): string => {

    return captureFrame(video);
};