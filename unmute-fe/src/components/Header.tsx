export function Header() {
  return (
    <header className="text-center mb-10 animate-fade-in-down">
      <div className="text-xs font-medium tracking-[0.25em] uppercase text-text-muted mb-3">
        Singapore Sign Language
      </div>
      <h1 className="font-serif text-[clamp(2.5rem,7vw,4.5rem)] font-semibold text-text-primary tracking-tight leading-none">
        un<span className="text-gradient">mute</span>
      </h1>
      <p className="font-sans text-base text-text-secondary mt-3 font-normal">
        Translate words into signs, effortlessly
      </p>
    </header>
  )
}
