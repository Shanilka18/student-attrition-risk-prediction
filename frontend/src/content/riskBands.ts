export interface RiskBandDefinition {
  min: number;
  max: number;
  label: string;
  hex_color?: string;
  action?: string;
}

export function getRiskBandDefinitionLabel(
  definition: RiskBandDefinition | undefined,
  fallbackLabel: string
): string {
  if (!definition) return fallbackLabel;
  return `${definition.label} (${definition.min}–${definition.max})`;
}
