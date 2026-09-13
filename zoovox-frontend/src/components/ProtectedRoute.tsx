import { Navigate, useLocation } from "react-router-dom";
import { token } from "@/services/api";

const ProtectedRoute = ({
  children,
}: {
  children: JSX.Element;
}) => {
  const location = useLocation();
  const accessToken = token.get();

  if (!accessToken) {
    return <Navigate to="/" replace state={{ from: location.pathname }} />;
  }

  return children;
};

export default ProtectedRoute;
