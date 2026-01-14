import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-[16px] font-medium transition-all duration-300 cursor-pointer disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default:
          "bg-gradient-accent text-white shadow-[0_4px_16px_rgba(198,123,92,0.3)] hover:translate-y-[-2px] hover:shadow-[0_8px_28px_rgba(198,123,92,0.4)] active:translate-y-0 active:scale-[0.98]",
        destructive:
          "bg-red-500 text-white shadow-sm hover:bg-red-600",
        outline:
          "border-[1.5px] border-border-warm bg-bg-card text-text-secondary hover:border-accent-terracotta hover:text-accent-terracotta hover:bg-accent-terracotta/5",
        secondary:
          "bg-bg-cream text-text-secondary border border-border-soft hover:bg-bg-card hover:text-text-primary",
        ghost: 
          "hover:bg-bg-cream hover:text-text-primary",
        link: 
          "text-accent-terracotta underline-offset-4 hover:underline",
      },
      size: {
        default: "h-11 px-6 py-2.5 text-sm",
        sm: "h-9 rounded-[12px] px-4 text-xs",
        lg: "h-12 rounded-[16px] px-8 text-base",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => {
    return (
      <button
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
