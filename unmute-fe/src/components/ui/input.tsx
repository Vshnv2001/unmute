import * as React from "react"

import { cn } from "@/lib/utils"

const Input = React.forwardRef<HTMLInputElement, React.ComponentProps<"input">>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-12 w-full rounded-[12px] border-[1.5px] border-border-soft bg-bg-card px-5 py-4 font-sans text-base text-text-primary placeholder:text-text-light focus:border-accent-terracotta focus:outline-none focus:ring-[3px] focus:ring-accent-terracotta/10 disabled:cursor-not-allowed disabled:opacity-50 transition-all duration-300",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Input.displayName = "Input"

export { Input }
