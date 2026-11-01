import React from "react";
import { useTranslation } from "react-i18next";
import { platform } from "@tauri-apps/plugin-os";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface LearnFromEditsProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

// The text-field readback uses Windows UI Automation, so it's Windows-only.
const isWindows = platform() === "windows";

export const LearnFromEdits: React.FC<LearnFromEditsProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    if (!isWindows) return null;

    return (
      <ToggleSwitch
        checked={getSetting("learn_from_edits") ?? false}
        onChange={(enabled) => updateSetting("learn_from_edits", enabled)}
        isUpdating={isUpdating("learn_from_edits")}
        label={t("settings.advanced.learnFromEdits.label")}
        description={t("settings.advanced.learnFromEdits.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
    );
  },
);
