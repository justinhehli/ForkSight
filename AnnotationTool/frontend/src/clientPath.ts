const isWindowsClient = (): boolean => {
  const uaData = (navigator as Navigator & { userAgentData?: { platform?: string } }).userAgentData;
  if (uaData?.platform) return uaData.platform.toLowerCase().includes("win");
  return /windows/i.test(navigator.userAgent);
};

export const formatPathForClipboard = (path: string): string => {
  const isNetworkShare = path.startsWith("//");
  return isNetworkShare && isWindowsClient() ? path.replace(/\//g, "\\") : path;
};
