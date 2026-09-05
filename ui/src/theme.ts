/**
 * Light or dark, chosen by the reader and kept.
 *
 * The OS preference is deliberately not consulted. The default is light
 * because that is what this is: an instrument read on a lit desk, closer to
 * a lab notebook than to an editor. The dark theme exists for the other
 * thing the screen gets used for - recording it - where a bright page blows
 * out the frame. Someone whose whole system is dark still gets the light
 * theme first, and one click changes it for good.
 *
 * `index.html` applies the stored value before the first paint; this module
 * is what writes it. Keep the key and the attribute in step with that script.
 */
import { useEffect, useState } from "react";

export type Theme = "light" | "dark";

export const THEME_KEY = "falsify.theme";

export function storedTheme(): Theme {
  try {
    return window.localStorage.getItem(THEME_KEY) === "dark" ? "dark" : "light";
  } catch {
    // a browser with site data blocked still gets a working application,
    // it just does not remember the choice
    return "light";
  }
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  try {
    window.localStorage.setItem(THEME_KEY, theme);
  } catch {
    /* nothing to do: the theme is applied, only the memory of it is lost */
  }
}

export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(storedTheme);
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);
  return [theme, () => setTheme((current) => (current === "dark" ? "light" : "dark"))];
}
