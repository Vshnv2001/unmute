import * as React from "react"

import { cn } from "@/lib/utils"

const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  React.ComponentProps<"textarea">
>(({ className, ...props }, ref) => {
  return (
    <textarea
      className={cn(
        "flex min-h-[130px] w-full rounded-[20px] border-[1.5px] border-border-soft bg-bg-input px-6 py-5 font-sans text-base text-text-primary leading-relaxed placeholder:text-text-light focus:border-accent-terracotta focus:bg-bg-card focus:outline-none focus:ring-4 focus:ring-accent-terracotta/10 disabled:cursor-not-allowed disabled:opacity-50 transition-all duration-300 resize-none",
        className
      )}
      ref={ref}
      {...props}
    />
  )
})
Textarea.displayName = "Textarea"

export { Textarea }
