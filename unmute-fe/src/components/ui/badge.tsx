import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-[10px] px-3.5 py-2 text-sm font-medium transition-all duration-300",
  {
    variants: {
      variant: {
        default:
          "bg-bg-cream border border-border-soft text-text-primary",
        active:
          "border border-accent-terracotta bg-accent-terracotta/10 text-accent-terracotta",
        missing:
          "border border-red-300/30 text-red-400/70 line-through",
        accent:
          "bg-gradient-accent text-white text-xs font-semibold tracking-wide",
        secondary:
          "bg-bg-cream text-text-secondary",
        outline: 
          "border border-border-soft text-text-secondary",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
